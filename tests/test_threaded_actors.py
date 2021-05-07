import pytest
from interlocking_backprop.threaded_actors import (
    ThreadedActor,
    create_future_with_value,
    on_worker_thread,
)


def test__on_worker_thread__method_throws_exception__exception_rethrown():
    class TestClass(ThreadedActor):
        @on_worker_thread
        def test_function(self) -> None:
            raise ValueError("Test exception")

    test_class = TestClass()
    result = test_class.test_function()

    with pytest.raises(ValueError, match="Test exception"):
        result.result()


def test__on_worker_thread__method_returns_value__future_contains_value():
    class TestClass(ThreadedActor):
        @on_worker_thread
        def test_function(self) -> str:
            return "test value"

    test_class = TestClass()
    result = test_class.test_function()

    assert result.result() == "test value"


def test__on_worker_thread__arguments_passed_correctly():
    class TestClass(ThreadedActor):
        @on_worker_thread
        def test_function(self, x: int, y: str = "3"):
            assert x == 4
            assert y == "10"

    test_class = TestClass()
    test_class.test_function(4, y="10").result()


def test__create_future_with_value__result_immediately_returns_value():
    future = create_future_with_value(3)
    assert future.result() == 3
