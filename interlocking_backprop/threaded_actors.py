"""Basic tools for an actor currency model, which uses a ThreadPoolExecutor under the hood."""
from __future__ import annotations

from abc import ABC
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


_THREAD_EXECUTOR_ATTRIBUTE = "_thread_executor"


def on_worker_thread(func: Callable[..., T]) -> Callable[..., Future[T]]:
    def _decorator(self, *args, **kwargs):
        if not hasattr(self, _THREAD_EXECUTOR_ATTRIBUTE):
            setattr(self, _THREAD_EXECUTOR_ATTRIBUTE, ThreadPoolExecutor(max_workers=1))

        thread_worker: ThreadPoolExecutor = getattr(self, _THREAD_EXECUTOR_ATTRIBUTE)
        return thread_worker.submit(func, self, *args, **kwargs)

    return _decorator


class ThreadedActor(ABC):
    @property
    def thread_executor(self) -> Optional[ThreadPoolExecutor]:
        return getattr(self, _THREAD_EXECUTOR_ATTRIBUTE, None)


def create_future_with_value(x: T) -> Future[T]:
    """Returns a Future which is immediately set with the given value."""
    # You're not meant to create futures directly, but I don't see another good way.
    future: Future[T] = Future()
    future.set_result(x)
    return future
