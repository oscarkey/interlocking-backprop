"""Builds ResNet models which can be used with InterlockingBackpropModel.

Based on: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from typing import List, Optional, Sequence, Tuple, Type, Union

from torch import Tensor, nn
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, MaxPool2d, Module, ReLU, Sequential
from torch.nn.modules.flatten import Flatten
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

Block = Union[BasicBlock, Bottleneck]
BlockType = Union[Type[BasicBlock], Type[Bottleneck]]
OutChannels = int


def resnet(
    block_type: str,
    architecture: str,
    aux_net_architecture: str,
    blocks_per_component: List[str],
    dataset: str,
    n_classes: int,
) -> Tuple[Sequence[nn.Module], Sequence[nn.Module]]:
    """Builds a ResNet and divides it into the specified number of components.

    :param block_type: either "basic" or "bottleneck", e.g. basic for ResNet-32, and bottleneck for ResNet-50
    :param archiecture: string is of the form: [block]-64,3/128,4/256,23/512,3 (this is ResNet-101) where slash
                        delimited list gives [n_planes],[n_blocks] for each superblock.
    :param aux_net_architecture: string which determines the architecture of the auxliliary networks, see
                                 build_aux_net().
    :param blocks_per_component: specifies how to devide the ResNet blocks into components. A comma-delimited list of
                                 the number of blocks in each component ending in 'rest' e.g. 3,4,rest (where rest
                                 indicates that all remaining blocks should go in the last component). The input layers
                                 together count as the first block, similary the output layers together count as the
                                 last block.
    """
    layers = _build_layers(dataset, architecture, n_classes, block_type)
    main_nets, aux_nets = _divide_layers_into_components(blocks_per_component, aux_net_architecture, n_classes, layers)

    for net in main_nets + aux_nets:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return main_nets, aux_nets


def _build_layers(
    dataset: str, architecture: str, n_classes: int, block_type: str
) -> List[Tuple[nn.Module, OutChannels]]:
    super_block_sizes = _parse_arch_string(architecture)
    block = _parse_block_type(block_type)

    layers = _build_input_layers(dataset, n_components=len(super_block_sizes))

    for superblock_i, (n_planes, n_blocks) in enumerate(super_block_sizes):
        stride = 1 if superblock_i == 0 else 2
        in_planes = layers[-1][1]
        layers.extend(_build_superblock_layers(in_planes, n_planes, n_blocks, stride, block))

    in_planes = layers[-1][1]
    layers.extend(_build_output_layers(in_planes, n_classes))

    return layers


def _divide_layers_into_components(
    blocks_per_component: List[str],
    aux_net_architecture: str,
    n_classes: int,
    layers: List[Tuple[nn.Module, OutChannels]],
) -> Tuple[List[nn.Module], List[nn.Module]]:
    main_nets: List[nn.Module] = []
    aux_nets: List[nn.Module] = []
    component_sizes = []
    next_index = 0
    for component_i, num_layers_str in enumerate(blocks_per_component):
        last_component = component_i + 1 == len(blocks_per_component)

        if last_component:
            if num_layers_str != "rest":
                raise ValueError('The layers_per_component string must end in "rest"')
            component_layers_and_out_planes = layers[next_index:]
        else:
            num_layers = int(num_layers_str)
            component_layers_and_out_planes = layers[next_index : next_index + num_layers]

            if len(component_layers_and_out_planes) == 0:
                raise ValueError("Component was specified as larger than number of blocks in model.")

        next_index += len(component_layers_and_out_planes)

        component_layers = [layer for layer, _ in component_layers_and_out_planes]
        main_nets.append(Sequential(*component_layers))

        if not last_component:
            _, in_channels = component_layers_and_out_planes[-1]
            aux_nets.append(build_aux_net(aux_net_architecture, in_channels, n_classes))

        component_sizes.append(len(component_layers_and_out_planes))

    print(f"Created ResNet of {len(component_sizes)} components of sizes {component_sizes}")

    return main_nets, aux_nets


def _parse_block_type(block_type: str) -> BlockType:
    if block_type == "basic":
        return BasicBlock
    elif block_type == "bottleneck":
        return Bottleneck
    else:
        raise ValueError(f"Unknown block type {block_type}")


def _parse_arch_string(arch_string: str) -> List[Tuple[int, int]]:
    """Returns (component type, block class, [(n_planes, n_blocks), ... for each super block])."""
    component_sizes = []
    for component_size in arch_string.split("/"):
        n_planes = int(component_size.split(",")[0])
        n_blocks = int(component_size.split(",")[1])
        component_sizes.append((n_planes, n_blocks))
    return component_sizes


class AssertCifar10Shape(Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.size()[1:4] == (3, 32, 32)
        return x


def _build_input_layers(dataset: str, n_components: int) -> List[Tuple[Module, OutChannels]]:
    num_planes = 64

    if dataset in ("cifar10", "cifar100"):
        assert n_components == 4, "We only support 4 components for CIFAR datasets."
        layers = [  #
            AssertCifar10Shape(),  #
            Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False),  #
            BatchNorm2d(num_planes),  #
            ReLU(),  #
        ]

    elif dataset == "imagenet":
        layers = [  #
            Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=False),  #
            BatchNorm2d(num_planes),  #
            ReLU(),  #
            MaxPool2d(kernel_size=3, stride=2, padding=1),  #
        ]

    else:
        raise ValueError(f"Unknown dataset {dataset}")

    return [(nn.Sequential(*layers), num_planes)]


def _build_output_layers(in_channels: int, n_classes: int) -> List[Tuple[Module, OutChannels]]:
    layers = [AdaptiveAvgPool2d((1, 1)), Flatten(), Linear(in_channels, n_classes)]
    return [(nn.Sequential(*layers), 1)]


def _build_superblock_layers(
    in_channels: int, n_planes: int, n_blocks: int, stride: int, block: BlockType
) -> Sequence[Tuple[Module, OutChannels]]:
    if stride != 1 or in_channels != n_planes * block.expansion:
        downsample: Optional[Module] = Sequential(
            conv1x1(in_channels, n_planes * block.expansion, stride), BatchNorm2d(n_planes * block.expansion),
        )
    else:
        downsample = None

    out_planes = n_planes * block.expansion
    layers = [
        (block(in_channels, n_planes, stride, downsample, norm_layer=BatchNorm2d), out_planes),
    ]
    for _ in range(1, n_blocks):
        layers.append((block(out_planes, n_planes, norm_layer=BatchNorm2d), out_planes))
    return layers


def build_aux_net(arch: str, in_channels: int, n_classes: int) -> nn.Sequential:
    """Builds the aux net from an architecture string of the form 'conv32_conv32_gbpl_fc'.

    This builds both the auxiliary networks for models with local loss functions, and the head of the entire network.

    Possible layers are: conv[channels] (convolutional), flt (flatten), gbpl (global pooling), fc (fully connected)
    """
    global_pooled = False
    layers: List[nn.Module] = []
    for layer in arch.split("_"):
        if layer.startswith("conv"):
            out_channels = int(layer.replace("conv", ""))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        elif layer == "bn":
            if isinstance(layers[-1], nn.ReLU):
                layers.pop()
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU())
        elif layer == "flt":
            raise NotImplementedError("As we are not using flattening, I removed it to reduce complexity.")
        elif layer == "gbpl":
            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())
            global_pooled = True
        elif layer == "fc":
            assert global_pooled, "Must global pool before fully connected"
            layers.append(nn.Linear(in_channels, n_classes))
        else:
            raise ValueError(f"Unknown aux net layer {layer}")
    return nn.Sequential(*layers)
