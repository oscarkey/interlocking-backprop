import math
from argparse import ArgumentParser
from typing import Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import Generator, Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

import interlocking_backprop
import interlocking_backprop.resnet_builder as resnet_builder
from interlocking_backprop.model import InterlockingBackpropModel


def main(dataset_root: str, mode: str):
    normalize_transform = Compose([ToTensor(), Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2460, 0.2411, 0.2576))])
    augment_transform = Compose([RandomHorizontalFlip(), RandomCrop(32, padding=4)])
    train_dataset = CIFAR10(dataset_root, train=True, transform=Compose([augment_transform, normalize_transform]))
    validation_dataset = CIFAR10(dataset_root, train=True, transform=normalize_transform)
    validation_length = int(math.floor(len(train_dataset) * 0.10))
    train_length = len(train_dataset) - validation_length
    train_dataset, _ = random_split(
        train_dataset, lengths=[train_length, validation_length], generator=Generator().manual_seed(0)
    )
    _, validation_dataset = random_split(
        validation_dataset, lengths=[train_length, validation_length], generator=Generator().manual_seed(0)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    if torch.cuda.device_count() == 0:
        device: Optional[torch.device] = torch.device("cpu")
        blocks_per_component = ["rest"]
    elif torch.cuda.device_count() == 1:
        device = torch.device("cuda")
        blocks_per_component = ["rest"]
    else:
        device = None
        # For the demo, just put one block on each device, and all remaing blocks on the last device.
        blocks_per_component = ["1"] * (torch.cuda.device_count() - 1) + ["rest"]

    # block_type and architecture correspond to ResNet-32.
    main_nets, aux_nets = resnet_builder.resnet(
        block_type="basic",
        architecture="64,3/128,4/256,6/512,3",
        aux_net_architecture="conv128_bn_conv64_bn_gbpl_fc",
        blocks_per_component=blocks_per_component,
        dataset="cifar10",
        n_classes=10,
    )

    optimizer_constructor = lambda params: SGD(params, lr=0.1, momentum=0.9, weight_decay=2e-4)
    # Learning rate schedule for ResNet-50ish on CIFAR10, taken from :
    # https://github.com/tensorflow/models/blob/master/official/r1/resnet/cifar10_main.py#L217
    lr_scheduler_constructor = lambda optimizer: MultiStepLR(optimizer, milestones=[91, 136, 182], gamma=0.1)
    loss_function = F.cross_entropy

    if mode == "e2e":
        model = interlocking_backprop.build_e2e_model(
            main_nets, optimizer_constructor, lr_scheduler_constructor, loss_function
        )
    elif mode == "local":
        model = interlocking_backprop.build_local_model(
            main_nets, aux_nets, optimizer_constructor, lr_scheduler_constructor, loss_function
        )
    elif mode == "pairwise":
        model = interlocking_backprop.build_pairwise_model(
            main_nets, aux_nets, optimizer_constructor, lr_scheduler_constructor, loss_function
        )
    elif mode == "nwise":
        model = interlocking_backprop.build_pairwise_model(
            main_nets, aux_nets, optimizer_constructor, lr_scheduler_constructor, loss_function
        )
    else:
        raise ValueError(f"Unknown mode {mode}")

    if torch.cuda.device_count() > 1:
        model.enable_model_parallel()
    else:
        model = model.to(device)

    print(f"Epoch 0: validation accuracy = {_compute_accuracy(validation_dataloader, model):.2f}")

    for epoch in range(100):
        model.train()
        losses = []
        for inputs, targets in train_dataloader:
            loss = model.training_step(inputs, targets)
            losses.append(loss)
        train_loss = torch.stack([loss.result() for loss in losses], axis=0).mean().item()
        validation_accuracy = _compute_accuracy(validation_dataloader, model)
        print(f"Epoch {epoch + 1}: training loss = {train_loss:.3f} validation accuracy = {validation_accuracy:.2f}")


def _compute_accuracy(dataloader: DataLoader, model: InterlockingBackpropModel) -> float:
    with torch.no_grad():
        model.eval()

        n_correct = 0
        n_total = 0
        for inputs, targets in dataloader:
            targets = targets.to(model.get_output_device())
            logits = model(inputs)
            batch_n_correct, batch_n_total = _compute_num_logits_correct(logits, targets)
            n_correct += batch_n_correct
            n_total += batch_n_total

        return n_correct / n_total


def _compute_num_logits_correct(logits: Tensor, targets: Tensor) -> Tuple[int, int]:
    """Returns (num logits correct, total size of batch)."""
    pred_labels = logits.argmax(dim=1)
    correct = pred_labels == targets
    n_correct = cast(int, torch.sum(correct).item())
    n_total = correct.size(0)
    return n_correct, n_total


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--mode", choices=["e2e", "local", "pairwise"], required=True)
    args = parser.parse_args()

    main(args.dataset_root, args.mode)
