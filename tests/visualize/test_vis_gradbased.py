# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from monai.visualize import GuidedBackpropGrad, GuidedBackpropSmoothGrad
from monai.visualize.gradient_based import _GuidedActivation

ACTIVATIONS = [nn.ReLU(), nn.LeakyReLU(negative_slope=0.2), nn.SiLU(), nn.GELU(), nn.GELU(approximate="tanh")]
ACTIVATION_NAMES = [type(act).__name__ for act in ACTIVATIONS]


def _local_derivative(act: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Compute the local derivative of an activation module at ``z``.

    The local derivative is ``dy / dz`` with ``y = act(z)`` and is used as the reference for the
    guided backpropagation rule.

    Args:
        act: activation module.
        z: point at which to evaluate the local derivative.

    Returns:
        Local derivative tensor with the same shape as ``z``.
    """
    z = z.detach().clone().requires_grad_(True)
    y = act(z)
    return torch.autograd.grad(y, z, torch.ones_like(y))[0]


class TestGuidedActivation(unittest.TestCase):
    """Tests for ``_GuidedActivation``: the forward pass is preserved and the backward pass follows the
    guided backpropagation rule (positive gradient only where the local derivative is positive)."""

    @parameterized.expand(list(zip(ACTIVATION_NAMES, ACTIVATIONS)))
    def test_forward_preserved(self, _name, act):
        """The guided wrapper's forward output equals the wrapped activation's output."""
        x = torch.randn(2, 3, 4)
        self.assertTrue(torch.allclose(_GuidedActivation(act)(x), act(x)))

    @parameterized.expand(list(zip(ACTIVATION_NAMES, ACTIVATIONS)))
    def test_backward_rule(self, _name, act):
        """The guided gradient equals ``grad_output * (grad_output > 0) * (local_derivative > 0)``."""
        grad_output = torch.randn(2, 3, 4)
        x = torch.randn(2, 3, 4).requires_grad_(True)
        _GuidedActivation(act)(x).backward(grad_output)
        local_derivative = _local_derivative(act, x.detach())
        expected = grad_output * (grad_output > 0) * (local_derivative > 0)
        self.assertTrue(torch.allclose(x.grad, expected))

    @parameterized.expand([("leaky_relu_inplace", nn.LeakyReLU(negative_slope=0.2, inplace=True))])
    def test_inplace_activation_preserves_aliasing(self, _name, act):
        """An in-place activation mutates its input (aliasing preserved) and the guided gradient is
        computed from the saved pre-activation values."""
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, requires_grad=True)
        x_pre = x.detach().clone()

        # forward: the in-place call mutates the input and the output holds the mutated values
        out = _GuidedActivation(act)(x)
        self.assertTrue(torch.allclose(x, act(x_pre.clone())))
        self.assertTrue(torch.allclose(out, x))

        # backward: guided rule applied to the saved pre-activation input
        grad_output = torch.randn(2, 3, 4)
        x2 = x_pre.clone().requires_grad_(True)
        _GuidedActivation(act)(x2).backward(grad_output)
        local_derivative = _local_derivative(nn.LeakyReLU(negative_slope=0.2), x_pre)
        expected = grad_output * (grad_output > 0) * (local_derivative > 0)
        self.assertTrue(torch.allclose(x2.grad, expected))


class TestGuidedBackpropGrad(unittest.TestCase):
    """Tests for ``GuidedBackpropGrad``: forward outputs are unchanged, modules are restored after the
    guided backward pass, and the guided rule is applied at every activation binding."""

    @parameterized.expand(list(zip(ACTIVATION_NAMES, ACTIVATIONS)))
    def test_forward_and_restore(self, _name, act):
        """Forward outputs are unchanged by the guided backward pass and the original modules are restored."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), act, nn.Linear(4, 3))
        before = model(torch.ones(1, 4)).detach().clone()
        module_types_before = [type(m) for _, m in model.named_modules()]

        x = torch.rand(1, 4)
        grad = GuidedBackpropGrad(model)(x, index=0)

        self.assertTupleEqual(grad.shape, x.shape)
        # forward outputs are unchanged by the guided backward pass
        self.assertTrue(torch.allclose(model(torch.ones(1, 4)).detach(), before))
        # the original activation modules are restored after execution
        self.assertEqual(module_types_before, [type(m) for _, m in model.named_modules()])

    def test_shared_activation_instance(self):
        """One activation instance bound twice must be wrapped at every binding; with
        ``named_modules`` dedup only the first binding would be wrapped and the gradient would be
        ``0.2`` instead of ``1`` for negative inputs."""
        torch.manual_seed(0)
        shared = nn.LeakyReLU(negative_slope=0.2)
        model = nn.Sequential(shared, shared)
        x = -torch.rand(1, 4)  # strictly negative inputs, LeakyReLU local derivative is 0.2

        grad = GuidedBackpropGrad(model)(x, index=0)

        # both bindings apply the guided rule, so the gradient passes through unchanged (local deriv > 0)
        self.assertTrue(torch.allclose(grad[0, 0], torch.tensor(1.0)))

    @parameterized.expand(list(zip(ACTIVATION_NAMES, ACTIVATIONS)))
    def test_guided_rule(self, _name, act):
        """The guided gradient through a ``Sequential(linear, act)`` matches the rule applied at the
        activation."""
        torch.manual_seed(1)
        linear = nn.Linear(5, 4)
        x = torch.randn(1, 5)
        class_idx = 1

        with torch.enable_grad():
            xi = x.clone().requires_grad_(True)
            zi = linear(xi)
            ai = act(zi)
            local_derivative = torch.autograd.grad(ai, zi, torch.ones_like(ai))[0]
        incoming = torch.zeros_like(ai)
        incoming[0, class_idx] = 1.0
        guided_at_act = incoming * (incoming > 0) * (local_derivative > 0)
        expected = torch.autograd.grad(zi, xi, guided_at_act)[0]

        model = nn.Sequential(linear, act)
        grad = GuidedBackpropGrad(model)(x.clone(), index=class_idx)
        self.assertTrue(torch.allclose(grad, expected, atol=1e-6))

    def test_smoothgrad_compatible(self):
        """``GuidedBackpropSmoothGrad`` runs end-to-end and returns a gradient with the input shape."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 3))
        x = torch.rand(1, 4)
        grad = GuidedBackpropSmoothGrad(model, n_samples=2, verbose=False)(x, index=0)
        self.assertTupleEqual(grad.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
