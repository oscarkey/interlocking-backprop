"""Wrappers around the `queue` module for when the queue items contain data on GPUs."""

from __future__ import annotations

import queue
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

import torch
from torch import Tensor

# The maximum time to wait for any thread, to avoid sitting in deadlock.
MAX_WAIT_SECS = 10


class DataOnGpu(ABC):
    """Indicates that a class contains Tensors on the GPU.

    This data will need to be moved from one device to another.
    """

    @abstractmethod
    def to(self: S, device: torch.device) -> S:
        """Returns a copy of this object with all Tensors copied to the given device."""
        pass

    def detach(self: S) -> S:
        """Returns a copy of this object with all Tensors detached from the graph."""
        # Subclasses only have to implement detach() if they expect to be detached.
        raise NotImplementedError


S = TypeVar("S", bound=DataOnGpu)

Data = Union[Tensor, DataOnGpu]
T = TypeVar("T", bound=Data)


class GpuAwareQueue(ABC, Generic[T]):
    """A wrapper around queue.Queue which has additional GPU related features:

     - get() moves any Tensors in the item to the specified GPU
     - put() can block until the consumer is ready to receive the item, and any Tensors
       in the item have been moved onto the correct GPU. This blocks the producer from
       loading any new Tensors onto the device until the old ones have been removed.
    """

    @abstractmethod
    def put(self, x: T, block_until_recieved: bool = True) -> None:
        """Adds the given item to the back of the queue.

        :param block_until_recieved: If True, this method will block until the item is
                                     removed by the consumer using get(). Regardless,
                                     this method will block if the queue is full.
        """
        pass

    @abstractmethod
    def get(self, device: torch.device) -> T:
        pass


class OneItemQueue(GpuAwareQueue[T]):
    """A GPU aware queue that has max size one."""

    def __init__(self, name: str = "unamed") -> None:
        self._queue: queue.Queue[T] = queue.Queue(maxsize=1)
        self._name = name

    def put(self, x: T, block_until_recieved: bool = True) -> None:
        # We always block until there is room in the queue. block_until_recieved
        # controls whether we block until the consumer has removed the item from the
        # queue. If True, then we block until task_done() is called in get(). At this
        # point the item has been moved on the correct device, and handed to the
        # consumer.
        self._queue.put(x, block=True)
        if block_until_recieved:
            self._queue.join()

    def get(self, device: torch.device) -> T:
        item = self._queue.get(block=True)
        item = item.to(device)
        self._queue.task_done()
        return item


class NoOpQueue(GpuAwareQueue[T]):
    """A queue that throws away items you put(). get() throws NotImplementedError."""

    def put(self, x: T, block_until_recieved: bool = True) -> None:
        # Just discard it.
        pass

    def get(self, device: torch.device) -> T:
        raise NotImplementedError(
            "Cannot get from a queue that does not store anything."
        )
