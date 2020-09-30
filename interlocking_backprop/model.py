from __future__ import annotations

import functools
import itertools
from concurrent.futures import Future
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter
from torch.optim.optimizer import Optimizer

from interlocking_backprop.components import (
    Component,
    ComponentQueues,
    E2EBackwardData,
    E2EComponent,
    ForwardData,
    LocalBackwardData,
    LocalLossOnlyComponent,
    LrScheduler,
    NwiseComponent,
    PairwiseComponent,
)
from interlocking_backprop.queues import MAX_WAIT_SECS, GpuAwareQueue, NoOpQueue, OneItemQueue
from interlocking_backprop.threaded_actors import ThreadedActor, create_future_with_value, on_worker_thread

LossFunction = Callable[[Tensor, Tensor], Tensor]
OptimizerConstructor = Callable[[Iterable[Parameter]], Optimizer]
LrSchedulerConstructor = Callable[[Optimizer], Optional[LrScheduler]]


class InterlockingBackpropModel(Module):
    def __init__(self, components: Iterable[Component], loss_function: LossFunction):
        super().__init__()
        self.components = ModuleList(components)
        self._component_queues = self._create_queues(components)
        self._loss_worker = _LossWorker(loss_function, self._component_queues[-1], self.components[-1])
        self._model_parallel = False

    @staticmethod
    def _create_queues(components: Iterable[Component]) -> List[ComponentQueues]:
        forward_in_queue: GpuAwareQueue[Tensor] = OneItemQueue(name="component_0_forward_in")
        # We don't care about the backward output of the final component, so use a queue which discards it.
        backward_out_queue: GpuAwareQueue[LocalBackwardData] = NoOpQueue()

        queue_holders = []
        for i, component in enumerate(components):
            queue_holder = ComponentQueues(component, forward_in_queue, backward_out_queue, name=f"component_{i}")
            queue_holders.append(queue_holder)

            forward_in_queue = queue_holder.forward_out_queue
            backward_out_queue = queue_holder.backward_in_queue

        return queue_holders

    def enable_model_parallel(self) -> None:
        self._model_parallel = True
        if len(self.components) > torch.cuda.device_count():
            raise ValueError(
                f"More components than GPUs " f"({len(self.components)} components {torch.cuda.device_count()} GPUs)."
            )
        for i, component in enumerate(self.components):
            component.to(f"cuda:{i}")

    def forward(self, x: ForwardData) -> ForwardData:
        """Performs a forward pass, and blocks until complete."""
        activations = x.detach()
        for component in self.components:
            activations = component(activations).detach()
        return activations

    def training_step(self, inputs: Tensor, targets: Tensor) -> Future[Tensor]:
        """Performs a step on a single batch.

        If in model parallel mode then this method will return before the training step is complete, to allow the next
        step to be queued. Otherwise, it blocks until all components have completed both the forward and backward pass.

        :param inputs: input Tensor, preferably on CPU
        :param targets: target Tensor, preferably on CPU
        :returns: a single item Tensor containing the loss, on the output device
        """
        # component.training_step() may block if the component hasn't finished its last step. Thus we put the inputs
        # into the queue first, so the first component can begin the next forward pass while the other components finish
        # the previous backward pass. Additionally, we set block_until_recieved=False so that forward_in_queue.put()
        # does not hang waiting for the first component to remove the data from the input queue, which only happens
        # when we call component[0].training_step().
        self._component_queues[0].forward_in_queue.put(inputs, block_until_recieved=False)
        results = [
            component.training_step(targets, queues)
            for component, queues in zip(self.components, self._component_queues)
        ]

        loss = self._loss_worker.compute_loss(targets)

        # If we're not in model parallel mode we want to block until all components are done, so wait on the futures.
        if not self._model_parallel:
            # We get the loss future to rethrow any exception. If the loss worker has thrown an exception,
            # the components may be in deadlock resulting in result.get() below timing out. Thus the caller will never
            # call loss.get(), and the exception in the loss worker will be hidden forever.
            loss_value = loss.result(timeout=MAX_WAIT_SECS)
            loss = create_future_with_value(loss_value)

            for result in results:
                # If the component encountered any exceptions, these will be rethrown here.
                result.result(timeout=MAX_WAIT_SECS)

        return loss

    def lr_scheduler_step(self, validation_loss: float) -> None:
        for component in self.components:
            component.lr_scheduler_step(validation_loss)

    def get_lr(self) -> float:
        # We assume that all components are using the same learning rate (which should be true), so just ask the first.
        return self.components[0].get_lr()

    def shutdown(self) -> None:
        for component in self.components:
            executor = component.thread_executor
            if executor is not None:
                executor.shutdown()

    def main_net_parameters(self) -> Iterator[Parameter]:
        return itertools.chain(*[component.main_net_parameters() for component in self.components])

    def get_input_device(self) -> torch.device:
        """Returns the device which input Tensors should be placed on to match the first component of the model."""
        return self.components[0].device

    def get_output_device(self) -> torch.device:
        """Returns the device which output Tensors will be placed on, matching the last component of the model."""
        return self.components[-1].device


class _LossWorker(ThreadedActor):
    def __init__(self, function: LossFunction, last_queues: ComponentQueues, last_component: Component) -> None:
        self._function = function
        self._forward_out_queue = last_queues.forward_out_queue
        self._backward_in_queue = last_queues.backward_in_queue
        self._last_component = last_component

    @on_worker_thread
    def compute_loss(self, targets: Tensor) -> Tensor:
        """Returns the detached loss."""
        targets = targets.to(self._last_component.device, non_blocking=True)
        logits = self._forward_out_queue.get(self._last_component.device)
        loss = self._function(logits, targets)
        (logits_gradients,) = torch.autograd.grad(loss, logits)

        backward_data = E2EBackwardData(logits_gradients)
        self._backward_in_queue.put(backward_data)

        return loss.detach()


def build_e2e_model(
    main_nets: Sequence[Module],
    optimizer_constructor: OptimizerConstructor,
    lr_scheduler_constructor: LrSchedulerConstructor,
    loss_function: LossFunction,
) -> InterlockingBackpropModel:
    optimizers = [optimizer_constructor(net.parameters()) for net in main_nets]
    lr_schedulers = [lr_scheduler_constructor(optimizer) for optimizer in optimizers]
    components = [
        E2EComponent(net, optimizer, lr_scheduler)
        for net, optimizer, lr_scheduler in zip(main_nets, optimizers, lr_schedulers)
    ]
    return InterlockingBackpropModel(components, loss_function)


def build_nwise_model(
    main_nets: Sequence[Module],
    aux_nets: Sequence[Module],
    optimizer_constructor: OptimizerConstructor,
    lr_scheduler_constructor: LrSchedulerConstructor,
    loss_function: LossFunction,
    nwise_communication_distance: int,
) -> InterlockingBackpropModel:
    component_constructor = functools.partial(NwiseComponent, nwise_communication_distance=nwise_communication_distance)
    return _build_aux_net_model(
        component_constructor, main_nets, aux_nets, optimizer_constructor, lr_scheduler_constructor, loss_function
    )


def build_pairwise_model(
    main_nets: Sequence[Module],
    aux_nets: Sequence[Module],
    optimizer_constructor: OptimizerConstructor,
    lr_scheduler_constructor: LrSchedulerConstructor,
    loss_function: LossFunction,
) -> InterlockingBackpropModel:
    return _build_aux_net_model(
        PairwiseComponent, main_nets, aux_nets, optimizer_constructor, lr_scheduler_constructor, loss_function
    )


def build_local_model(
    main_nets: Sequence[Module],
    aux_nets: Sequence[Module],
    optimizer_constructor: OptimizerConstructor,
    lr_scheduler_constructor: LrSchedulerConstructor,
    loss_function: LossFunction,
) -> InterlockingBackpropModel:
    return _build_aux_net_model(
        LocalLossOnlyComponent, main_nets, aux_nets, optimizer_constructor, lr_scheduler_constructor, loss_function
    )


def _build_aux_net_model(
    component_constructor: Callable[[Module, Module, Optimizer, Optional[LrScheduler]], Component],
    main_nets: Sequence[Module],
    aux_nets: Sequence[Module],
    optimizer_constructor: OptimizerConstructor,
    lr_scheduler_constructor: LrSchedulerConstructor,
    loss_function: LossFunction,
) -> InterlockingBackpropModel:
    if len(aux_nets) != len(main_nets) - 1:
        raise ValueError("Incorrect aux nets")

    components: List[Component] = []
    for main_net, aux_net in zip(main_nets, aux_nets):
        optimizer = optimizer_constructor(itertools.chain(main_net.parameters(), aux_net.parameters()))
        components.append(component_constructor(main_net, aux_net, optimizer, lr_scheduler_constructor(optimizer)))

    # main_nets has one extra element than aux_nets, which we put in the final component.
    optimizer = optimizer_constructor(main_nets[-1].parameters())
    components.append(E2EComponent(main_nets[-1], optimizer, lr_scheduler_constructor(optimizer)))

    return InterlockingBackpropModel(components, loss_function)
