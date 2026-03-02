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
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from parameterized import parameterized

from monai.engines import GradientAccumulation
from monai.utils import IgniteInfo, min_version, optional_import
from monai.utils.enums import CommonKeys

_, has_ignite = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version)

INVALID_ACCUMULATION_STEPS = [
    (0,), (-1,), (2.5,), ("2",),
]

SUPPRESSION_CASES = [
    # (attr_name, acc, epoch_length, num_iters, expected)
    (
        "zero_grad", 4, 12, 12,
        [True, False, False, False, True, False, False, False, True, False, False, False],
    ),
    (
        "step", 4, 12, 12,
        [False, False, False, True, False, False, False, True, False, False, False, True],
    ),
    # epoch_length=11 not divisible by 4 → flush at epoch end
    (
        "step", 4, 11, 11,
        [False, False, False, True, False, False, False, True, False, False, True],
    ),
    # epoch_length=None (iterable dataset) → no epoch flush
    (
        "step", 4, None, 10,
        [False, False, False, True, False, False, False, True, False, False],
    ),
]


def _make_engine(epoch_length, iteration=1, scaler=None):
    """Create a mock engine whose _iteration observes patched methods."""
    engine = MagicMock()
    engine.state.epoch_length = epoch_length
    engine.state.iteration = iteration
    engine.scaler = scaler
    engine.optimizer = MagicMock()
    engine.loss_function = MagicMock(return_value=torch.tensor(1.0))
    engine._iteration.return_value = {CommonKeys.LOSS: torch.tensor(1.0)}
    return engine


class TestGradientAccumulation(unittest.TestCase):
    """Test cases for GradientAccumulation callable."""

    # ---- input validation ----

    @parameterized.expand(INVALID_ACCUMULATION_STEPS)
    def test_invalid_accumulation_steps(self, value) -> None:
        with self.assertRaises(ValueError) as cm:
            GradientAccumulation(accumulation_steps=value)
        self.assertIn("positive integer", str(cm.exception))

    def test_repr(self) -> None:
        ga = GradientAccumulation(accumulation_steps=8)
        self.assertEqual(repr(ga), "GradientAccumulation(accumulation_steps=8)")

    # ---- passthrough ----

    def test_passthrough_when_accumulation_steps_1(self) -> None:
        grad_accum = GradientAccumulation(accumulation_steps=1)
        engine = _make_engine(epoch_length=12, iteration=1)
        expected_output = {CommonKeys.LOSS: torch.tensor(0.5), CommonKeys.PRED: torch.tensor([1.0])}
        engine._iteration.return_value = expected_output

        result = grad_accum(engine, {})

        engine._iteration.assert_called_once_with(engine, {})
        self.assertIs(result, expected_output)

    # ---- suppression logic ----

    @parameterized.expand(SUPPRESSION_CASES)
    def test_suppression(self, attr_name, acc, epoch_length, num_iters, expected) -> None:
        grad_accum = GradientAccumulation(accumulation_steps=acc)
        original = MagicMock(name=attr_name)
        engine = _make_engine(epoch_length)
        setattr(engine.optimizer, attr_name, original)

        saw_original: list[bool] = []

        def fake_iteration(eng, batch):
            saw_original.append(getattr(eng.optimizer, attr_name) is original)
            return {CommonKeys.LOSS: torch.tensor(1.0)}

        engine._iteration.side_effect = fake_iteration

        for i in range(1, num_iters + 1):
            engine.state.iteration = i
            grad_accum(engine, {})

        self.assertEqual(saw_original, expected)

    # ---- patching / restoration ----

    def test_patching_and_restoration(self) -> None:
        engine = _make_engine(epoch_length=4, iteration=1)

        original_zero_grad = MagicMock(name="original_zero_grad")
        original_step = MagicMock(name="original_step")
        original_loss_fn = MagicMock(return_value=torch.tensor(0.5), name="original_loss_fn")

        engine.optimizer.zero_grad = original_zero_grad
        engine.optimizer.step = original_step
        engine.loss_function = original_loss_fn

        GradientAccumulation(accumulation_steps=2)(engine, {})

        self.assertIs(engine.optimizer.zero_grad, original_zero_grad)
        self.assertIs(engine.optimizer.step, original_step)
        self.assertIs(engine.loss_function, original_loss_fn)

    def test_restoration_after_exception(self) -> None:
        """try/finally must restore all originals even when _iteration raises."""
        engine = _make_engine(epoch_length=8, iteration=2)

        original_zero_grad = MagicMock(name="zero_grad")
        original_step = MagicMock(name="step")
        original_loss_fn = MagicMock(return_value=torch.tensor(1.0), name="loss_fn")
        original_scaler_step = MagicMock(name="scaler_step")
        original_scaler_update = MagicMock(name="scaler_update")

        engine.optimizer.zero_grad = original_zero_grad
        engine.optimizer.step = original_step
        engine.loss_function = original_loss_fn
        engine.scaler = MagicMock()
        engine.scaler.step = original_scaler_step
        engine.scaler.update = original_scaler_update
        engine._iteration.side_effect = RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            GradientAccumulation(accumulation_steps=4)(engine, {})

        self.assertIs(engine.optimizer.zero_grad, original_zero_grad)
        self.assertIs(engine.optimizer.step, original_step)
        self.assertIs(engine.loss_function, original_loss_fn)
        self.assertIs(engine.scaler.step, original_scaler_step)
        self.assertIs(engine.scaler.update, original_scaler_update)

    # ---- scaler ----

    def test_scaler_not_patched_when_stepping(self) -> None:
        engine = _make_engine(epoch_length=4, iteration=2)  # acc=2 → should_step=True
        original_scaler_step = MagicMock(name="scaler_step")
        original_scaler_update = MagicMock(name="scaler_update")
        engine.scaler = MagicMock()
        engine.scaler.step = original_scaler_step
        engine.scaler.update = original_scaler_update

        GradientAccumulation(accumulation_steps=2)(engine, {})

        self.assertIs(engine.scaler.step, original_scaler_step)
        self.assertIs(engine.scaler.update, original_scaler_update)

    def test_scaler_patched_and_restored_when_suppressed(self) -> None:
        engine = _make_engine(epoch_length=8, iteration=2)  # should_step=False for acc=4
        original_scaler_step = MagicMock(name="scaler_step")
        original_scaler_update = MagicMock(name="scaler_update")
        engine.scaler = MagicMock()
        engine.scaler.step = original_scaler_step
        engine.scaler.update = original_scaler_update

        scaler_was_patched = []

        def check_scaler(eng, batch):
            scaler_was_patched.append(eng.scaler.step is not original_scaler_step)
            return {CommonKeys.LOSS: torch.tensor(0.5)}

        engine._iteration.side_effect = check_scaler
        GradientAccumulation(accumulation_steps=4)(engine, {})

        self.assertTrue(scaler_was_patched[0], "scaler.step should be patched during _iteration")
        self.assertIs(engine.scaler.step, original_scaler_step)
        self.assertIs(engine.scaler.update, original_scaler_update)

    def test_no_scaler_attribute(self) -> None:
        """Engine without a scaler attribute at all should work (hasattr returns False)."""
        engine = _make_engine(epoch_length=4, iteration=1)
        del engine.scaler  # MagicMock auto-creates attrs; delete to test hasattr branch

        result = GradientAccumulation(accumulation_steps=2)(engine, {})
        self.assertIn(CommonKeys.LOSS, result)

    def test_scaler_is_none(self) -> None:
        engine = _make_engine(epoch_length=4, iteration=2)
        engine.scaler = None

        result = GradientAccumulation(accumulation_steps=2)(engine, {})
        self.assertIn(CommonKeys.LOSS, result)

    # ---- batch / loss ----

    def test_batch_data_passed_correctly(self) -> None:
        engine = _make_engine(epoch_length=4, iteration=1)
        test_batch = {CommonKeys.IMAGE: torch.randn(1, 10), CommonKeys.LABEL: torch.randn(1, 1)}

        GradientAccumulation(accumulation_steps=2)(engine, test_batch)

        engine._iteration.assert_called_once()
        call_args = engine._iteration.call_args
        self.assertEqual(call_args[0][0], engine)
        self.assertEqual(call_args[0][1], test_batch)

    def test_loss_output_is_unscaled(self) -> None:
        """Output loss should be rescaled to the original (unscaled) value."""
        engine = _make_engine(epoch_length=9, iteration=1)
        engine.scaler = None
        original_loss = torch.tensor(6.0)
        engine.loss_function = MagicMock(return_value=original_loss)

        def fake_iteration(*args, **kwargs):
            scaled = engine.loss_function()
            return {CommonKeys.LOSS: scaled}

        engine._iteration.side_effect = fake_iteration

        result = GradientAccumulation(accumulation_steps=3)(engine, {})
        self.assertAlmostEqual(result[CommonKeys.LOSS].item(), 6.0, places=5)

    # ---- integration (require ignite) ----

    @unittest.skipUnless(has_ignite, "Requires pytorch-ignite")
    def test_integration_gradient_equivalence(self) -> None:
        """Accumulated gradients over N mini-batches equal one large-batch step."""
        from monai.engines import SupervisedTrainer

        torch.manual_seed(42)
        acc_steps, lr = 4, 0.1
        batches = [
            {CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)}
            for _ in range(acc_steps)
        ]

        ref_model, test_model, ref_opt, test_opt, init_weight = _make_model_pair(lr)

        ref_opt.zero_grad()
        for batch in batches:
            loss = nn.MSELoss()(ref_model(batch[CommonKeys.IMAGE]), batch[CommonKeys.LABEL]).mean() / acc_steps
            loss.backward()
        ref_opt.step()

        trainer = SupervisedTrainer(
            device=torch.device("cpu"), max_epochs=1, train_data_loader=batches,
            network=test_model, optimizer=test_opt, loss_function=nn.MSELoss(),
            iteration_update=GradientAccumulation(accumulation_steps=acc_steps),
        )
        trainer.run()

        for p_test, p_ref in zip(test_model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(p_test.data, p_ref.data)

    @unittest.skipUnless(has_ignite, "Requires pytorch-ignite")
    def test_integration_epoch_boundary_flush(self) -> None:
        """When epoch_length is not divisible by acc_steps, flush at epoch end."""
        from monai.engines import SupervisedTrainer

        torch.manual_seed(123)
        acc_steps, lr = 3, 0.1
        batches = [
            {CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)}
            for _ in range(5)
        ]

        ref_model, test_model, ref_opt, test_opt, init_weight = _make_model_pair(lr)

        for cycle_batches in [batches[:3], batches[3:]]:
            ref_opt.zero_grad()
            for batch in cycle_batches:
                loss = nn.MSELoss()(ref_model(batch[CommonKeys.IMAGE]), batch[CommonKeys.LABEL]).mean() / acc_steps
                loss.backward()
            ref_opt.step()

        trainer = SupervisedTrainer(
            device=torch.device("cpu"), max_epochs=1, train_data_loader=batches,
            network=test_model, optimizer=test_opt, loss_function=nn.MSELoss(),
            iteration_update=GradientAccumulation(accumulation_steps=acc_steps),
        )
        trainer.run()

        for p_test, p_ref in zip(test_model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(p_test.data, p_ref.data)

    @unittest.skipUnless(has_ignite, "Requires pytorch-ignite")
    def test_integration_multi_epoch(self) -> None:
        """Verify gradient accumulation is correct across multiple epochs."""
        from monai.engines import SupervisedTrainer

        torch.manual_seed(42)
        acc_steps, lr, num_epochs = 2, 0.1, 3
        batches = [
            {CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)}
            for _ in range(4)
        ]

        ref_model, test_model, ref_opt, test_opt, init_weight = _make_model_pair(lr)

        for _epoch in range(num_epochs):
            for cycle_batches in [batches[:2], batches[2:]]:
                ref_opt.zero_grad()
                for batch in cycle_batches:
                    loss = nn.MSELoss()(ref_model(batch[CommonKeys.IMAGE]), batch[CommonKeys.LABEL]).mean() / acc_steps
                    loss.backward()
                ref_opt.step()

        trainer = SupervisedTrainer(
            device=torch.device("cpu"), max_epochs=num_epochs, train_data_loader=batches,
            network=test_model, optimizer=test_opt, loss_function=nn.MSELoss(),
            iteration_update=GradientAccumulation(accumulation_steps=acc_steps),
        )
        trainer.run()

        for p_test, p_ref in zip(test_model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(p_test.data, p_ref.data)


def _make_model_pair(lr):
    """Create a reference and test model pair with identical initial weights."""
    ref_model = nn.Linear(4, 1, bias=False)
    init_weight = ref_model.weight.data.clone()
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=lr)
    ref_model.train()

    test_model = nn.Linear(4, 1, bias=False)
    test_model.weight.data.copy_(init_weight)
    test_opt = torch.optim.SGD(test_model.parameters(), lr=lr)

    return ref_model, test_model, ref_opt, test_opt, init_weight


if __name__ == "__main__":
    unittest.main()
