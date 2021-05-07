from typing import Callable, Iterable

import interlocking_backprop.resnet_builder as resnet_builder
import pytest
import torch
import torch.nn.functional as F
from interlocking_backprop import build_e2e_model
from interlocking_backprop.model import InterlockingBackpropModel
from numpy.random.mtrand import RandomState
from torch.nn import AdaptiveAvgPool2d, Linear, Module, Parameter
from torch.nn.modules import Flatten
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.optim import SGD, Adam
from torch.optim.optimizer import Optimizer
from torchvision.models import resnet34, resnet50, resnet101

OPTIMIZER_CONSTRUCTOR = lambda params: SGD(params, lr=1.0)
LR_SCHED_CONSTURCTOR = lambda optimizer: None
LOSS_FUNC = F.cross_entropy


def test__resnet__101__forward_pass_equal_to_torchvision_resnet():
    # The torchvision ResNet is the ImageNet implementation, hence use the ImageNet
    # version of our implementation.
    our_resnet_main_nets, _ = resnet_builder.resnet(
        block_type="bottleneck",
        architecture="64,3/128,4/256,23/512,3",
        aux_net_architecture="gbpl_fc",
        blocks_per_component=["2", "3", "rest"],
        dataset="imagenet",
        n_classes=10,
    )
    our_resnet = build_e2e_model(
        our_resnet_main_nets, OPTIMIZER_CONSTRUCTOR, LR_SCHED_CONSTURCTOR, LOSS_FUNC
    ).double()
    torchvision_resnet = resnet101(num_classes=10).double()

    # Ensure the models have the same parameter values.
    for our_param, torchvision_param in zip(
        our_resnet.parameters(), torchvision_resnet.parameters()
    ):
        assert (
            our_param.size() == torchvision_param.size()
        ), f"not same {our_param} {torchvision_param}"
        our_param.data = torchvision_param.data.clone()

    test_input = torch.tensor(RandomState(seed=0).randn(10, 3, 32, 32)).double()
    torchvision_output = torchvision_resnet(test_input)
    our_output = our_resnet(test_input)

    assert torch.allclose(our_output, torchvision_output)


def test__resnet__50__forward_pass_equal_to_torchvision_resnet():
    # The torchvision ResNet is the ImageNet implementation, hence use the ImageNet
    # version of our implementation.
    our_resnet_main_nets, _ = resnet_builder.resnet(
        block_type="bottleneck",
        architecture="64,3/128,4/256,6/512,3",
        aux_net_architecture="gbpl_fc",
        blocks_per_component=["2", "3", "rest"],
        dataset="imagenet",
        n_classes=10,
    )
    our_resnet = build_e2e_model(
        our_resnet_main_nets, OPTIMIZER_CONSTRUCTOR, LR_SCHED_CONSTURCTOR, LOSS_FUNC
    ).double()
    torchvision_resnet = resnet50(num_classes=10).double()

    # Ensure the models have the same parameter values.
    for our_param, torchvision_param in zip(
        our_resnet.parameters(), torchvision_resnet.parameters()
    ):
        assert (
            our_param.size() == torchvision_param.size()
        ), f"not same {our_param} {torchvision_param}"
        our_param.data = torchvision_param.data.clone()

    test_input = torch.tensor(RandomState(seed=0).randn(10, 3, 32, 32)).double()
    torchvision_output = torchvision_resnet(test_input)
    our_output = our_resnet(test_input)

    assert torch.allclose(our_output, torchvision_output)


def test__resnet__50__logits_equal_after_several_iterations():
    def optimizer_constructor(params):
        return Adam(params)

    # The torchvision ResNet is the ImageNet implementation, hence use the ImageNet
    # version of our implementation.
    our_resnet_main_nets, _ = resnet_builder.resnet(
        block_type="bottleneck",
        architecture="64,3/128,4/256,6/512,3",
        aux_net_architecture="gbpl_fc",
        blocks_per_component=["2", "3", "rest"],
        dataset="imagenet",
        n_classes=10,
    )
    our_resnet = build_e2e_model(
        our_resnet_main_nets, optimizer_constructor, LR_SCHED_CONSTURCTOR, LOSS_FUNC
    ).double()
    torchvision_resnet = resnet50(num_classes=10).double()
    _assert_models_optimize_equivalently(
        our_resnet, torchvision_resnet, optimizer_constructor
    )


def test__resnet__34__logits_equal_after_several_iterations():
    def optimizer_constructor(params):
        return Adam(params)

    # The torchvision ResNet is the ImageNet implementation, hence use the ImageNet
    # version of our implementation.
    our_resnet_main_nets, _ = resnet_builder.resnet(
        block_type="basic",
        architecture="64,3/128,4/256,6/512,3",
        aux_net_architecture="gbpl_fc",
        blocks_per_component=["2", "3", "rest"],
        dataset="imagenet",
        n_classes=10,
    )
    our_resnet = build_e2e_model(
        our_resnet_main_nets, optimizer_constructor, LR_SCHED_CONSTURCTOR, LOSS_FUNC
    ).double()
    torchvision_resnet = resnet34(num_classes=10).double()
    _assert_models_optimize_equivalently(
        our_resnet, torchvision_resnet, optimizer_constructor
    )


def _assert_models_optimize_equivalently(
    component_model: InterlockingBackpropModel,
    torchvision_model: Module,
    optimizer_constructor: Callable[[Iterable[Parameter]], Optimizer],
) -> None:
    # Ensure the models have the same parameter values.
    assert len(list(component_model.parameters())) == len(
        list(torchvision_model.parameters())
    )
    for our_param, torchvision_param in zip(
        component_model.parameters(), torchvision_model.parameters()
    ):
        assert (
            our_param.size() == torchvision_param.size()
        ), f"not same {our_param} {torchvision_param}"
        our_param.data = torchvision_param.data.clone()

    torchvision_optimizer = optimizer_constructor(torchvision_model.parameters())

    random = RandomState(seed=0)
    for i in range(3):
        inputs = torch.tensor(random.randn(2, 3, 32, 32)).double()
        targets = torch.tensor([0, 3])

        our_logits = component_model(inputs.clone())
        component_model.training_step(inputs.clone(), targets.clone())

        torchvision_logits = torchvision_model(inputs.clone())
        torchvision_optimizer.zero_grad()
        torchvision_loss = F.cross_entropy(torchvision_logits, targets.clone())
        torchvision_loss.backward()
        torchvision_optimizer.step()

        assert torch.allclose(
            torchvision_logits, our_logits
        ), f"Outputs different at iteration {i}"


def test__build_aux_net__gblpl_fc__returns_correct_network():
    net = resnet_builder.build_aux_net(arch="gbpl_fc", in_channels=3, n_classes=10)

    x = torch.randn((10, 3, 16, 16))
    y = net(x)
    assert y.size() == (10, 10)

    layers = net.children()
    assert isinstance(next(layers), AdaptiveAvgPool2d)
    assert isinstance(next(layers), Flatten)
    assert isinstance(next(layers), Linear)


@pytest.mark.skip(reason="Flattening is not current supported, we werent using it.")
def test__build_aux_net__flt_fc__returns_correct_network():
    net = resnet_builder.build_aux_net(arch="flt_fc", in_channels=9, n_classes=10)

    x = torch.randn((12, 9, 8, 8))
    y = net(x)
    assert y.size() == (12, 10)

    layers = net.children()
    assert isinstance(next(layers), Flatten)
    assert isinstance(next(layers), Linear)


def test__build_aux_net__conv_gblfl_fc__returns_correct_network():
    net = resnet_builder.build_aux_net(
        arch="conv32_gbpl_fc", in_channels=3, n_classes=10
    )

    x = torch.randn((20, 3, 16, 16))
    y = net(x)
    assert y.size() == (20, 10)

    layers = net.children()
    assert isinstance(next(layers), Conv2d)
    assert isinstance(next(layers), ReLU)
    assert isinstance(next(layers), AdaptiveAvgPool2d)
    assert isinstance(next(layers), Flatten)
    assert isinstance(next(layers), Linear)


def test__build_aux_net__conv_bn_gblfl_fc__returns_correct_network():
    net = resnet_builder.build_aux_net(
        arch="conv32_bn_gbpl_fc", in_channels=3, n_classes=10
    )

    x = torch.randn((20, 3, 16, 16))
    y = net(x)
    assert y.size() == (20, 10)

    layers = net.children()
    assert isinstance(next(layers), Conv2d)
    assert isinstance(next(layers), BatchNorm2d)
    assert isinstance(next(layers), ReLU)
    assert isinstance(next(layers), AdaptiveAvgPool2d)
    assert isinstance(next(layers), Flatten)
    assert isinstance(next(layers), Linear)
