import itertools
from typing import Iterable, Iterator, List

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.optim import SGD

from interlocking_backprop.components import E2EComponent, PairwiseComponent
from interlocking_backprop.model import (
    InterlockingBackpropModel,
    PairwiseComponent,
    build_e2e_model,
    build_local_model,
    build_nwise_model,
    build_pairwise_model,
)


class _TestNet(Module):
    """Simple module with easy to calculate gradients."""

    def __init__(self, param: List[float]) -> None:
        super().__init__()
        self.param = Parameter(torch.tensor(param))

    def forward(self, x: Tensor) -> Tensor:
        return self.param * x


def _create_test_optimizer(params: Iterable[Parameter]) -> SGD:
    # This just subtracts the gradient from the current parameter value.
    return SGD(params, lr=1.0)


OPTIMIZER_CONSTRUCTOR = lambda params: SGD(params, lr=1.0)
LR_SCHED_CONSTURCTOR = lambda optimizer: None
LOSS_FUNC = F.cross_entropy


class FakeLossFunction:
    """A loss function which allows control over the gradient of the logits.

    Returns a loss value which, when differentiated, results in gradient of the logits
    being as specified. Being able to specify the logit gradient makes it easier to
    manually work out the gradients of the rest of the model.
    """

    def __init__(self, logit_grad: float) -> None:
        self._logit_grad = logit_grad

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Return a Tensor with the desired gradient.
        return torch.sum(logits * self._logit_grad)


class TestComponentModel:
    def test__e2e__backward_pass__net_parameters_updated_correctly(self):
        main_nets = [
            _TestNet([2.0]),
            _TestNet([3.0]),
            _TestNet([4.0]),
        ]
        model = build_e2e_model(
            main_nets,
            OPTIMIZER_CONSTRUCTOR,
            LR_SCHED_CONSTURCTOR,
            FakeLossFunction(logit_grad=0.1),
        )

        inputs = torch.tensor([[1.0, 0.0]])
        targets = torch.tensor([1])
        model.training_step(inputs, targets).result()

        # We expect the gradient to be propagated fully through the model.
        # The gradients are (worked out by hand):
        # wrt logits=0.1, wrt c3=0.6, wrt c2=0.8, wrt c1=1.2
        assert torch.allclose(next(main_nets[2].parameters()), torch.tensor([[3.4]]))
        assert torch.allclose(next(main_nets[1].parameters()), torch.tensor([[2.2]]))
        assert torch.allclose(next(main_nets[0].parameters()), torch.tensor([[0.8]]))

    def test__e2e__several_iterations__output_equal_to_simple_model(self):
        def optimizer_constructor(params):
            return SGD(params, lr=0.01)

        main_nets = [
            Sequential(Linear(2, 3), ReLU()),
            Sequential(Linear(3, 3), ReLU()),
            Sequential(Linear(3, 3), ReLU()),
            Sequential(Linear(3, 2)),
        ]
        component_model = build_e2e_model(
            main_nets, optimizer_constructor, LR_SCHED_CONSTURCTOR, LOSS_FUNC
        )

        simple_model = Sequential(
            Linear(2, 3),
            ReLU(),
            Linear(3, 3),
            ReLU(),
            Linear(3, 3),
            ReLU(),
            Linear(3, 2),
        )

        # Ensure the models have the same parameter values.
        for component_param, simple_param in zip(
            component_model.parameters(), simple_model.parameters()
        ):
            assert (
                component_param.size() == simple_param.size()
            ), f"not same {component_param} {simple_param}"
            component_param.data = simple_param.data.clone()

        optimizer_simple = optimizer_constructor(simple_model.parameters())

        for i in range(10):
            x = torch.tensor([[0.2, 0.45]]) * (i + 1)
            targets = torch.tensor([0])

            logits_component = component_model(x.clone())
            component_model.training_step(x.clone(), targets.clone()).result()

            logits_simple = simple_model(x.clone())
            optimizer_simple.zero_grad()
            loss_simple = F.cross_entropy(logits_simple, targets.clone())
            loss_simple.backward()
            optimizer_simple.step()

            assert torch.allclose(
                logits_simple, logits_component
            ), f"Outputs different at iteration {i}"

    def test__local__backward_pass__main_net_parameters_updated_correctly(self):
        main_nets = [
            _TestNet([2.0, 2.0]),
            _TestNet([3.0, 3.0]),
            _TestNet([4.0, 4.0]),
        ]
        aux_nets = [
            _TestNet([1.0, 1.0]),
            _TestNet([1.0, 1.0]),
        ]
        model = build_local_model(
            main_nets,
            aux_nets,
            OPTIMIZER_CONSTRUCTOR,
            LR_SCHED_CONSTURCTOR,
            FakeLossFunction(logit_grad=0.1),
        )

        # We have to use 2 dimensional data so we can have incorrect logits, and thus a
        # non-zero loss and gradient.
        model.training_step(
            inputs=torch.tensor([[1.0, 1.0]]), targets=torch.tensor([1])
        ).result()

        # Component 3: gradients based on gradients of the true logits,
        #              giving [0.6, 0.6].
        # Component 2: gradients based on local loss, giving [1., -1.]
        # Component 1: gradients based on local loss, giving [0.5, -0.5]
        assert torch.allclose(
            next(main_nets[2].parameters()), torch.tensor([[3.4, 3.4]])
        )
        assert torch.allclose(
            next(main_nets[1].parameters()), torch.tensor([[2.0, 4.0]])
        )
        assert torch.allclose(
            next(main_nets[0].parameters()), torch.tensor([[1.5, 2.5]])
        )

    def test__pairwise_only__backward_pass__main_net_parameters_updated_correctly(self):
        main_nets = [
            _TestNet([2.0, 2.0]),
            _TestNet([3.0, 3.0]),
            _TestNet([4.0, 4.0]),
        ]
        aux_nets = [
            _TestNet([1.0, 1.0]),
            _TestNet([1.0, 1.0]),
        ]
        model = build_pairwise_model(
            main_nets,
            aux_nets,
            OPTIMIZER_CONSTRUCTOR,
            LR_SCHED_CONSTURCTOR,
            FakeLossFunction(logit_grad=0.1),
        )

        # We have to use 2 dimensional data so we can have incorrect logits, and thus a
        # non-zero loss and gradient.
        model.training_step(
            inputs=torch.tensor([[1.0, 1.0]]), targets=torch.tensor([1])
        ).result()

        # Component 3: gradients based on gradients of the true logits,
        #              giving [0.6, 0.6].
        # Component 2: gradients based on the gradients from component 3,
        #              giving [0.8, 0.8]
        # Component 1: gradients based on gradients from local loss of component 2,
        #              giving [1.5, -1.5]
        assert torch.allclose(
            next(main_nets[2].parameters()), torch.tensor([[3.4, 3.4]])
        )
        assert torch.allclose(
            next(main_nets[1].parameters()), torch.tensor([[2.2, 2.2]])
        )
        assert torch.allclose(
            next(main_nets[0].parameters()), torch.tensor([[0.5, 3.5]])
        )

    @pytest.mark.skip(
        reason="Currently pairwise only supports 'pairwise only', so disable this test."
    )
    def test__pairwise_sum__backward_pass__main_net_parameters_updated_correctly(self):
        # Currently pairwise only supports "pairwise only", so disable this test.
        # config = test_config(pairwise_grad_mode="sum")
        components = [
            PairwiseComponent(_TestNet([2.0, 2.0]), _TestNet([1.0, 1.0]), config),
            PairwiseComponent(_TestNet([3.0, 3.0]), _TestNet([1.0, 1.0]), config),
            E2EComponent(_TestNet([4.0, 4.0]), config),
        ]
        model = InterlockingBackpropModel(config, components)
        optimizer = _create_test_optimizer(model.parameters())

        # We have to use 2 dimensional data so we can have incorrect logits, and thus a
        # non-zero loss and gradient.
        y = model(torch.tensor([[1.0, 1.0]]))
        optimizer.zero_grad()
        model.backward_pass(
            logits_gradients=torch.tensor([[0.1, 0.1]]), true_targets=torch.tensor([1])
        )
        optimizer.step()

        # Component 3: grads based on grads of the true logits, giving [0.6, 0.6].
        # Component 2: grads based on the grads from component 3 and local loss,
        #              giving [0.8+1.0, 0.8+-1.0] = [1.8, -0.2]
        # Component 1: grads based on grads from local loss of 2 and local loss of 1,
        #              giving [1.5+0.5, -1.5+-0.5]
        # = [2.0, -2.0]
        assert torch.allclose(
            next(components[2].parameters()), torch.tensor([[3.4, 3.4]])
        )
        assert torch.allclose(
            next(components[1].parameters()), torch.tensor([[1.2, 3.2]])
        )
        assert torch.allclose(
            next(components[0].parameters()), torch.tensor([[0.0, 4.0]])
        )

    @pytest.mark.skip(
        reason="Currently pairwise only supports 'pairwise only', so disable this test."
    )
    def test__pairwise_mean__backward_pass__main_net_parameters_updated_correctly(self):
        # config = test_config(pairwise_grad_mode="mean")
        components = [
            PairwiseComponent(_TestNet([2.0, 2.0]), _TestNet([1.0, 1.0]), config),
            PairwiseComponent(_TestNet([3.0, 3.0]), _TestNet([1.0, 1.0]), config),
            E2EComponent(_TestNet([4.0, 4.0]), config),
        ]
        model = InterlockingBackpropModel(config, components)
        optimizer = _create_test_optimizer(model.parameters())

        # We have to use 2 dimensional data so we can have incorrect logits, and thus a
        # non-zero loss and gradient.
        y = model(torch.tensor([[1.0, 1.0]]))
        optimizer.zero_grad()
        model.backward_pass(
            logits_gradients=torch.tensor([[0.1, 0.1]]), true_targets=torch.tensor([1])
        )
        optimizer.step()

        # Component 3: grads based on grads of the true logits, giving [0.6, 0.6].
        # Component 2: grads based on the grads from component 3 and local loss,
        #              giving [mean(0.8, 1.0), mean(0.8, -1.0)] = [0.9, -0.1]
        # Component 1: grads based on grads from local loss of 2 and local loss of 1,
        #              giving [mean(1.5, 0.5), mean(-1.5, -0.5)] = [1.0, -1.0]
        assert torch.allclose(
            next(components[2].parameters()), torch.tensor([[3.4, 3.4]])
        )
        assert torch.allclose(
            next(components[1].parameters()), torch.tensor([[2.1, 3.1]])
        )
        assert torch.allclose(
            next(components[0].parameters()), torch.tensor([[1.0, 3.0]])
        )

    def test__nwise_distance_1__backward_pass__parameters_equal_to_pairwise_only(self):
        pairwise_main_nets = [
            _TestNet([2.0, 2.0]),
            _TestNet([3.0, 3.0]),
            _TestNet([4.0, 4.0]),
            _TestNet([5.0, 5.0]),
        ]
        pairwise_aux_nets = [
            _TestNet([1.0, 1.0]),
            _TestNet([1.0, 1.0]),
            _TestNet([1.0, 1.0]),
        ]
        pairwise_model = build_pairwise_model(
            pairwise_main_nets,
            pairwise_aux_nets,
            OPTIMIZER_CONSTRUCTOR,
            LR_SCHED_CONSTURCTOR,
            FakeLossFunction(logit_grad=0.1),
        )

        nwise_main_nets = [
            _TestNet([2.0, 2.0]),
            _TestNet([3.0, 3.0]),
            _TestNet([4.0, 4.0]),
            _TestNet([5.0, 5.0]),
        ]
        nwise_aux_nets = [
            _TestNet([1.0, 1.0]),
            _TestNet([1.0, 1.0]),
            _TestNet([1.0, 1.0]),
        ]
        nwise_model = build_nwise_model(
            nwise_main_nets,
            nwise_aux_nets,
            OPTIMIZER_CONSTRUCTOR,
            LR_SCHED_CONSTURCTOR,
            FakeLossFunction(logit_grad=0.1),
            nwise_communication_distance=1,
        )

        pairwise_params_start = [param.clone() for param in pairwise_model.parameters()]
        nwise_params_start = [param.clone() for param in nwise_model.parameters()]

        x = torch.tensor([[1.0, 1.0]])
        pairwise_y = pairwise_model(x.clone())
        nwise_y = nwise_model(x.clone())

        targets = torch.tensor([1])

        pairwise_model.training_step(x.clone(), targets.clone()).result()
        nwise_model.training_step(x.clone(), targets.clone()).result()

        pairwise_params_end = [param.clone() for param in pairwise_model.parameters()]
        nwise_params_end = [param.clone() for param in nwise_model.parameters()]

        assert torch.allclose(pairwise_y, nwise_y)
        for pairwise_param, nwise_param in zip(
            pairwise_params_start, nwise_params_start
        ):
            print("pairwise", pairwise_params_end)
            assert torch.allclose(pairwise_param, nwise_param)
        for pairwise_param, nwise_param in zip(
            pairwise_params_end[1:], nwise_params_end[1:]
        ):
            assert torch.allclose(pairwise_param, nwise_param)

    def test__nwise_distance_1__several_iterations__output_equal_to_pairwise(self):
        def optimizer_constructor(params):
            return SGD(params, lr=0.01)

        pairwise_main_nets = [
            Linear(2, 3),
            Linear(3, 3),
            Linear(3, 3),
            Linear(3, 3),
            Linear(3, 2),
        ]
        pairwise_aux_nets = [
            Linear(3, 2),
            Linear(3, 2),
            Linear(3, 2),
            Linear(3, 2),
        ]
        pairwise_model = build_pairwise_model(
            pairwise_main_nets,
            pairwise_aux_nets,
            optimizer_constructor,
            LR_SCHED_CONSTURCTOR,
            LOSS_FUNC,
        )

        nwise_main_nets = [
            Linear(2, 3),
            Linear(3, 3),
            Linear(3, 3),
            Linear(3, 3),
            Linear(3, 2),
        ]
        nwise_aux_nets = [
            Linear(3, 2),
            Linear(3, 2),
            Linear(3, 2),
            Linear(3, 2),
        ]
        nwise_model = build_nwise_model(
            nwise_main_nets,
            nwise_aux_nets,
            optimizer_constructor,
            LR_SCHED_CONSTURCTOR,
            LOSS_FUNC,
            nwise_communication_distance=1,
        )

        # Ensure the models have the same parameter values.
        for pairwise_param, nwise_param in zip(
            pairwise_model.parameters(), nwise_model.parameters()
        ):
            assert (
                pairwise_param.size() == nwise_param.size()
            ), f"not same {pairwise_param} {nwise_param}"
            nwise_param.data = pairwise_param.data.clone()

        for i in range(10):
            x = torch.tensor([[0.5, 0.4]]) * (i + 1)
            targets = torch.tensor([0])

            logits_pairwise = pairwise_model(x.clone())
            pairwise_model.training_step(x.clone(), targets.clone()).result()

            logits_nwise = nwise_model(x.clone())
            nwise_model.training_step(x.clone(), targets.clone()).result()

            assert torch.allclose(
                logits_pairwise, logits_nwise
            ), f"Outputs different at iteration {i}"

    def test__nwise_distance_all__several_iterations__output_equal_to_e2e(self):
        def optimizer_constructor(params):
            return SGD(params, lr=0.01)

        e2e_main_nets = [
            Linear(2, 3),
            Linear(3, 3),
            Linear(3, 3),
            Linear(3, 3),
            Linear(3, 2),
        ]
        e2e_model = build_e2e_model(
            e2e_main_nets, optimizer_constructor, LR_SCHED_CONSTURCTOR, LOSS_FUNC,
        )

        nwise_main_nets = [
            Linear(2, 3),
            Linear(3, 3),
            Linear(3, 3),
            Linear(3, 3),
            Linear(3, 2),
        ]
        nwise_aux_nets = [
            Linear(3, 2),
            Linear(3, 2),
            Linear(3, 2),
            Linear(3, 2),
        ]
        nwise_model = build_nwise_model(
            nwise_main_nets,
            nwise_aux_nets,
            optimizer_constructor,
            LR_SCHED_CONSTURCTOR,
            LOSS_FUNC,
            nwise_communication_distance=4,
        )

        # Ensure the models have the same parameter values.
        for e2e_param, nwise_param in zip(
            e2e_model.parameters(), self._get_main_net_params(nwise_model)
        ):
            assert (
                e2e_param.size() == nwise_param.size()
            ), f"not same {e2e_param} {nwise_param}"
            nwise_param.data = e2e_param.data.clone()

        for i in range(10):
            x = torch.tensor([[0.5, 0.4]]) * (i + 1)
            targets = torch.tensor([0])

            logits_e2e = e2e_model(x.clone())
            e2e_model.training_step(x.clone(), targets.clone()).result()

            logits_nwise = nwise_model(x.clone())
            nwise_model.training_step(x.clone(), targets.clone()).result()

            assert torch.allclose(
                logits_e2e, logits_nwise
            ), f"Outputs different at iteration {i}"

    @staticmethod
    def _get_main_net_params(model: InterlockingBackpropModel) -> Iterator[Parameter]:
        return itertools.chain(
            *[component._net.parameters() for component in model.components]
        )

    def test__nwise_distance_2__backward_pass__main_net_params_updated_correctly(self):
        main_nets = [
            _TestNet([2.0, 2.0]),
            _TestNet([3.0, 3.0]),
            _TestNet([4.0, 4.0]),
            _TestNet([5.0, 5.0]),
        ]
        aux_nets = [
            _TestNet([1.0, 1.0]),
            _TestNet([1.0, 1.0]),
            _TestNet([1.0, 1.0]),
        ]
        model = build_nwise_model(
            main_nets,
            aux_nets,
            OPTIMIZER_CONSTRUCTOR,
            LR_SCHED_CONSTURCTOR,
            FakeLossFunction(logit_grad=0.1),
            nwise_communication_distance=2,
        )

        model.training_step(
            inputs=torch.tensor([[1.0, 1.0]]), targets=torch.tensor([1])
        ).result()

        # Component 4: grads based on grads of true logits, [2.4, 2.4]
        # Component 3: grads based on propagated grads from component 4, [3.0, 3.0]
        # Component 2: grads based on propagated grads from component 4, [4.0, 4.0]
        # Component 1: grads based on propagated local grads from component 3, [6, -6]
        assert torch.allclose(next(main_nets[3].parameters()), torch.tensor([2.6, 2.6]))
        assert torch.allclose(next(main_nets[2].parameters()), torch.tensor([1.0, 1.0]))
        assert torch.allclose(
            next(main_nets[1].parameters()), torch.tensor([-1.0, -1.0])
        )
        assert torch.allclose(
            next(main_nets[0].parameters()), torch.tensor([-4.0, 8.0])
        )
