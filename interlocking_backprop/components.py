from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from interlocking_backprop.queues import DataOnGpu, GpuAwareQueue, NoOpQueue, OneItemQueue
from interlocking_backprop.threaded_actors import ThreadedActor, on_worker_thread

LrScheduler = Union[MultiStepLR, ReduceLROnPlateau]
ForwardData = Union[Tensor, DataOnGpu]


class LocalBackwardData(DataOnGpu):
    """Data propagated through the components during the backward pass.

    This class is empty because no data is communicated when using local communication. The NwiseBackwardData and
    E2EBackwardData subclasses contain additional data available when using greater communication between components.
    """

    def to(self, device: torch.device) -> LocalBackwardData:
        # We don't have any data, thus there is nothing to move.
        return LocalBackwardData()


class NwiseBackwardData(LocalBackwardData):
    """Data propagated through the components during the backward pass.

    This class contains data propagated when using nwise communication, including pairwise communication. It may contain
    data which depends on the backward passes of the next (n-1) components in the model.
    """

    def __init__(self, nwise_gradients: List[Tensor]) -> None:
        super().__init__()
        assert len(nwise_gradients) > 0
        self._nwise_gradients = nwise_gradients

    @property
    def nwise_gradients(self) -> List[Tensor]:
        """List of gradients propagated backwards through the model.

        The first gradient in the list corresponds to the loss of the next component in the model, and subsequent
        gradients to components closer to the head of the model.
        For a communication distance of (n-1), this list will contain at most (n-1) gradients. It may contain less if
        the backward pass is currently near the head of the model and has processed fewer than (n-1) components.
        """
        return self._nwise_gradients

    @property
    def pairwise_gradients(self) -> Tensor:
        """Gradients wrt the loss of the next component propagated to the previous component only.

        This is the first gradient in the nwise_gradients list.
        """
        return self._nwise_gradients[0]

    def to(self, device: torch.device) -> NwiseBackwardData:
        return NwiseBackwardData([grad.to(device) for grad in self._nwise_gradients])

    @staticmethod
    def create_from_pairwise_gradients(pairwise_gradients: Tensor) -> NwiseBackwardData:
        return NwiseBackwardData([pairwise_gradients])


class E2EBackwardData(NwiseBackwardData):
    """Data propagated through the components during the backward pass.

    This class contains data propagated when using pairwise communication. It may contain data which depends on the
    backward pass of all subsequent components in the model.
    """

    def __init__(self, e2e_gradients: Tensor) -> None:
        # The overall model may consist of several pairwise components followed by an e2e component as the final layer.
        # Thus set the pairwise gradients to the e2e gradients to allow any previous pairwise components to read them.
        super().__init__([e2e_gradients])
        self._e2e_gradients = e2e_gradients

    @property
    def e2e_gradients(self) -> Tensor:
        """Gradients wrt the overall loss propagated through all of the components."""
        return self._e2e_gradients

    def to(self, device: torch.device) -> E2EBackwardData:
        return E2EBackwardData(self._e2e_gradients.to(device))


class Component(Module, ThreadedActor, ABC):
    def __init__(self, net: Module, optimizer: Optimizer, lr_scheduler: Optional[LrScheduler]):
        super().__init__()
        self._net = net
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        self._log_scalars: Dict[str, List[float]] = defaultdict(lambda: [])

    def forward(self, x: ForwardData) -> ForwardData:
        x = x.to(self.device)
        return self._net(x)

    @on_worker_thread
    def training_step(self, targets: Tensor, queues: ComponentQueues) -> None:
        self.zero_grad()

        inputs = queues.forward_in_queue.get(self.device).detach()
        inputs.requires_grad = True
        activations = self(inputs)
        queues.forward_out_queue.put(activations)

        self._backward_pass(targets, queues.backward_in_queue, queues.backward_out_queue, inputs, activations)

        self._optimizer.step()

    @abstractmethod
    def _backward_pass(
        self,
        targets: Tensor,
        in_queue: GpuAwareQueue[LocalBackwardData],
        out_queue: GpuAwareQueue[LocalBackwardData],
        inputs: Tensor,
        activations: Tensor,
    ) -> None:
        pass

    @on_worker_thread
    def lr_scheduler_step(self, validation_loss: float) -> None:
        if self._optimizer is None:
            raise ValueError("Must call training_step at least once before steping lr scheduler.")

        if self._lr_scheduler is None:
            # This happens if the lr scheduler is disabled.
            return

        if isinstance(self._lr_scheduler, MultiStepLR):
            self._lr_scheduler.step()
        elif isinstance(self._lr_scheduler, ReduceLROnPlateau):
            self._lr_scheduler.step(validation_loss)
        else:
            raise ValueError

    def get_lr(self) -> float:
        lrs = [param_group["lr"] for param_group in self._optimizer.param_groups]
        assert all(lr == lrs[0] for lr in lrs), "Currently all param groups must use the same learning rate."
        return lrs[0]

    def get_and_reset_log_scalars(self) -> Dict[str, float]:
        """Returns the mean of each scalar log value since this method was last called."""
        data = {k: sum(v) / len(v) for k, v in self._log_scalars.items()}
        self._log_scalars = defaultdict(lambda: [])
        return data

    @property
    def device(self) -> torch.device:
        # This assumes all parameters are on the same device, which should be true.
        return next(self.parameters()).device

    def main_net_parameters(self) -> Iterator[Parameter]:
        """Returns the parameters of the main network i.e. excluding any auxliary network parameters."""
        return self._net.parameters()

    @property
    @abstractmethod
    def requires_backward_communication(self) -> bool:
        pass

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        self._extend_state_dict(destination, self._optimizer.state_dict(), prefix=prefix + "optimizer")
        if self._lr_scheduler is not None:
            self._extend_state_dict(destination, self._lr_scheduler.state_dict(), prefix=prefix + "lr_scheduler")

    @staticmethod
    def _extend_state_dict(destination: Dict, additional: Dict, prefix: str) -> None:
        for k, v in additional.items():
            destination[prefix + "." + k] = v

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        optimizer_state, state_dict = self._retrieve_prefix_from_state_dict(state_dict, prefix=prefix + "optimizer")
        self._optimizer.load_state_dict(optimizer_state)
        if self._lr_scheduler is not None:
            lr_scheduler_state, state_dict = self._retrieve_prefix_from_state_dict(
                state_dict, prefix=prefix + "lr_scheduler"
            )
            self._lr_scheduler.load_state_dict(lr_scheduler_state)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @staticmethod
    def _retrieve_prefix_from_state_dict(state_dict: Dict, prefix: str) -> Tuple[Dict, Dict]:
        retrieved = {}
        original = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                retrieved[k.replace(prefix + ".", "")] = v
            else:
                original[k] = v
        return retrieved, original


class E2EComponent(Component):
    """Component which allows communication of gradients through the entire model."""

    def _backward_pass(
        self,
        targets: Tensor,
        in_queue: GpuAwareQueue[LocalBackwardData],
        out_queue: GpuAwareQueue[LocalBackwardData],
        inputs: Tensor,
        activations: Tensor,
    ) -> None:
        backward_data = cast(E2EBackwardData, in_queue.get(self.device))
        activations.backward(gradient=backward_data.e2e_gradients)
        out_queue.put(E2EBackwardData(inputs.grad))

    @property
    def requires_backward_communication(self) -> bool:
        return True


class AuxNetComponent(Component, ABC):
    """Abstract class for any component which has an auxiliary network used during training."""

    def __init__(self, net: Module, aux_net: Module, optimizer: Optimizer, lr_scheduler: Optional[LrScheduler]) -> None:
        super().__init__(net, optimizer, lr_scheduler)
        self._aux_net = aux_net
        self._last_activations: Optional[ForwardData] = None

    def forward(self, x: ForwardData) -> ForwardData:
        self._last_activations = super().forward(x)
        return self._last_activations

    def eval_aux_net(self) -> Tensor:
        """Evaluates the auxiliary network of this component on the activations from the last forward pass.

        :returns: the auxiliary logits
        """
        if self._last_activations is None:
            raise ValueError("Must perform a forward pass first.")
        return self._aux_net(self._last_activations)


class LocalLossOnlyComponent(AuxNetComponent):
    def __init__(self, net: Module, aux_net: Module, optimizer: Optimizer, lr_scheduler: Optional[LrScheduler]) -> None:
        super().__init__(net, aux_net, optimizer, lr_scheduler)

    def _backward_pass(
        self,
        targets: Tensor,
        in_queue: GpuAwareQueue[LocalBackwardData],
        out_queue: GpuAwareQueue[LocalBackwardData],
        inputs: Tensor,
        activations: Tensor,
    ) -> None:
        # As we rely on the local loss, we don't get or put any data.
        targets = targets.to(self.device, non_blocking=True)
        aux_logits = self._aux_net(activations)
        aux_loss = F.cross_entropy(aux_logits, targets)
        aux_loss.backward()

    @property
    def requires_backward_communication(self) -> bool:
        return False


class PairwiseComponent(AuxNetComponent):
    def _backward_pass(
        self,
        targets: Tensor,
        in_queue: GpuAwareQueue[LocalBackwardData],
        out_queue: GpuAwareQueue[LocalBackwardData],
        inputs: Tensor,
        activations: Tensor,
    ) -> None:
        # There are 5 gradients involved:
        # 1) Gradient of the local loss function wrt our inputs (which are the activations of the previous component)
        # 2) Gradient of the local loss function wrt the main network parameters
        # 3) Gradient of the local loss function wrt the auxiliary network parameters
        # 4) Gradient of the next component's loss function wrt our activations
        # 5) Gradient of the next component's loss function wrt our main network parameters

        targets = targets.to(self.device, non_blocking=True)

        # Compute (3) and use it to update the aux network parameters.
        aux_net_inputs = activations.detach()
        aux_net_inputs.requires_grad = True
        self._aux_net.zero_grad()
        aux_logits = self._aux_net(aux_net_inputs)
        aux_loss = F.cross_entropy(aux_logits, targets)
        aux_loss.backward()

        if not isinstance(out_queue, NoOpQueue):
            # This means we have a component behind us, so we need to compute (1) and pass it to it, and update the aux
            # net parameters.
            (input_grads_from_local_loss,) = torch.autograd.grad(
                activations, inputs, grad_outputs=aux_net_inputs.grad, retain_graph=True
            )
            out_queue.put(NwiseBackwardData.create_from_pairwise_gradients(input_grads_from_local_loss))

        # Retrieve (4), use it to compute (5), and update the main network parameters with it.
        grads_from_next_component = cast(NwiseBackwardData, in_queue.get(self.device)).pairwise_gradients
        self._net.zero_grad()
        activations.backward(gradient=grads_from_next_component)

    @property
    def requires_backward_communication(self) -> bool:
        return True


class NwiseComponent(AuxNetComponent):
    """Component which propagates gradients through n components.

    This component updates its parameters with gradients propagated from the loss function (n-1) components ahead in the
    model. Near the head of the model this is the true loss function, further back it is the local loss function.

    :param nwise_communication_distance: controls the number of components through which gradients will propagate.
                                         nwise_communication_distance = 1 is equivalent to pairwise_only
                                         nwise_communication_distance = (num components - 1) is equivalent to e2e
    """

    def __init__(
        self,
        net: Module,
        aux_net: Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[LrScheduler],
        nwise_communication_distance: int,
    ) -> None:
        super().__init__(net, aux_net, optimizer, lr_scheduler)

        if nwise_communication_distance < 1:
            raise ValueError(f"Communication distance must be at least 1")
        self._max_communication_distance = nwise_communication_distance

    def _backward_pass(
        self,
        targets: Tensor,
        in_queue: GpuAwareQueue[LocalBackwardData],
        out_queue: GpuAwareQueue[LocalBackwardData],
        inputs: Tensor,
        activations: Tensor,
    ) -> None:
        aux_net_inputs = activations.detach()
        aux_net_inputs.requires_grad = True
        aux_logits = self._aux_net(aux_net_inputs)
        aux_loss = F.cross_entropy(aux_logits, targets.to(self.device, non_blocking=True))
        aux_loss.backward()

        # Propagate gradients through this component
        backward_data = cast(NwiseBackwardData, in_queue.get(self.device))
        output_gradients = [aux_net_inputs.grad.detach()] + backward_data.nwise_gradients
        nwise_grads = []
        for i, output_grads in enumerate(output_gradients):
            # We want to update the main network parameters with only the gradients from the component
            # furthest away in the model, which are computed in the final iteration of the loop.
            last_iteration = i + 1 == len(output_gradients)
            if last_iteration:
                assert inputs.grad is None
                activations.backward(gradient=output_grads)
                grads = cast(Tensor, inputs.grad).detach()
            else:
                (grads,) = torch.autograd.grad(activations, inputs, grad_outputs=output_grads, retain_graph=True)

            nwise_grads.append(grads)

        trimmed_nwise_grads = nwise_grads[: self._max_communication_distance]
        out_queue.put(NwiseBackwardData(trimmed_nwise_grads))

    @property
    def requires_backward_communication(self) -> bool:
        return True


class ComponentQueues:
    def __init__(
        self,
        component: Component,
        forward_in_queue: GpuAwareQueue[Tensor],
        backward_out_queue: GpuAwareQueue[LocalBackwardData],
        name: str,
    ) -> None:
        self.forward_in_queue = forward_in_queue
        self.forward_out_queue: OneItemQueue[Tensor] = OneItemQueue(name=name + "_forward_out")

        if component.requires_backward_communication:
            self.backward_in_queue: GpuAwareQueue[LocalBackwardData] = OneItemQueue(name=name + "_backward_in")
        else:
            self.backward_in_queue = NoOpQueue()
        self.backward_out_queue = backward_out_queue
