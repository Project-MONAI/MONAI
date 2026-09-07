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

from monai.utils import IgniteInfo, min_version, optional_import
from monai.utils.enums import CommonKeys

_, has_ignite = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version)

INVALID_ACCUMULATION_STEPS = [(0,), (-1,)]


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


@unittest.skipUnless(has_ignite, "Requires pytorch-ignite")
class TestGradientAccumulation(unittest.TestCase):
    """Test gradient accumulation integrated into SupervisedTrainer."""

    # ---- input validation ----

    @parameterized.expand(INVALID_ACCUMULATION_STEPS)
    def test_invalid_accumulation_steps(self, value) -> None:
        from monai.engines import SupervisedTrainer

        with self.assertRaises(ValueError) as cm:
            SupervisedTrainer(
                device=torch.device("cpu"),
                max_epochs=1,
                train_data_loader=[{CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)}],
                network=nn.Linear(4, 1),
                optimizer=torch.optim.SGD(nn.Linear(4, 1).parameters(), lr=0.1),
                loss_function=nn.MSELoss(),
                accumulation_steps=value,
            )
        self.assertIn("positive integer", str(cm.exception))

    # ---- passthrough (accumulation_steps=1) ----

    def test_passthrough_when_accumulation_steps_1(self) -> None:
        """With accumulation_steps=1, behaviour is identical to default training."""
        from monai.engines import SupervisedTrainer

        torch.manual_seed(42)
        lr = 0.1
        batches = [{CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)} for _ in range(4)]

        ref_model, test_model, ref_opt, test_opt, _ = _make_model_pair(lr)

        # Reference: standard training loop
        for batch in batches:
            ref_opt.zero_grad()
            loss = nn.MSELoss()(ref_model(batch[CommonKeys.IMAGE]), batch[CommonKeys.LABEL]).mean()
            loss.backward()
            ref_opt.step()

        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=batches,
            network=test_model,
            optimizer=test_opt,
            loss_function=nn.MSELoss(),
            accumulation_steps=1,
        )
        trainer.run()

        for p_test, p_ref in zip(test_model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(p_test.data, p_ref.data)

    # ---- gradient equivalence ----

    def test_gradient_equivalence(self) -> None:
        """Accumulated gradients over N mini-batches equal one large-batch step."""
        from monai.engines import SupervisedTrainer

        torch.manual_seed(42)
        acc_steps, lr = 4, 0.1
        batches = [{CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)} for _ in range(acc_steps)]

        ref_model, test_model, ref_opt, test_opt, _ = _make_model_pair(lr)

        # Reference: manual accumulation
        ref_opt.zero_grad()
        for batch in batches:
            loss = nn.MSELoss()(ref_model(batch[CommonKeys.IMAGE]), batch[CommonKeys.LABEL]).mean() / acc_steps
            loss.backward()
        ref_opt.step()

        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=batches,
            network=test_model,
            optimizer=test_opt,
            loss_function=nn.MSELoss(),
            accumulation_steps=acc_steps,
        )
        trainer.run()

        for p_test, p_ref in zip(test_model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(p_test.data, p_ref.data)

    # ---- epoch boundary flush ----

    def test_epoch_boundary_flush(self) -> None:
        """When epoch_length is not divisible by acc_steps, flush at epoch end."""
        from monai.engines import SupervisedTrainer

        torch.manual_seed(123)
        acc_steps, lr = 3, 0.1
        batches = [{CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)} for _ in range(5)]

        ref_model, test_model, ref_opt, test_opt, _ = _make_model_pair(lr)

        # Reference: first 3 batches form one cycle, last 2 form a partial cycle flushed at epoch end
        for cycle_batches in [batches[:3], batches[3:]]:
            ref_opt.zero_grad()
            for batch in cycle_batches:
                loss = nn.MSELoss()(ref_model(batch[CommonKeys.IMAGE]), batch[CommonKeys.LABEL]).mean() / acc_steps
                loss.backward()
            ref_opt.step()

        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=batches,
            network=test_model,
            optimizer=test_opt,
            loss_function=nn.MSELoss(),
            accumulation_steps=acc_steps,
        )
        trainer.run()

        for p_test, p_ref in zip(test_model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(p_test.data, p_ref.data)

    # ---- multi-epoch ----

    def test_multi_epoch(self) -> None:
        """Verify gradient accumulation is correct across multiple epochs."""
        from monai.engines import SupervisedTrainer

        torch.manual_seed(42)
        acc_steps, lr, num_epochs = 2, 0.1, 3
        batches = [{CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)} for _ in range(4)]

        ref_model, test_model, ref_opt, test_opt, _ = _make_model_pair(lr)

        # Reference: manual multi-epoch accumulation
        for _epoch in range(num_epochs):
            for cycle_batches in [batches[:2], batches[2:]]:
                ref_opt.zero_grad()
                for batch in cycle_batches:
                    loss = nn.MSELoss()(ref_model(batch[CommonKeys.IMAGE]), batch[CommonKeys.LABEL]).mean() / acc_steps
                    loss.backward()
                ref_opt.step()

        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=num_epochs,
            train_data_loader=batches,
            network=test_model,
            optimizer=test_opt,
            loss_function=nn.MSELoss(),
            accumulation_steps=acc_steps,
        )
        trainer.run()

        for p_test, p_ref in zip(test_model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(p_test.data, p_ref.data)

    # ---- loss output is unscaled ----

    def test_loss_output_is_unscaled(self) -> None:
        """engine.state.output[LOSS] should be the unscaled loss, not loss/acc."""
        from monai.engines import SupervisedTrainer

        torch.manual_seed(42)
        acc_steps = 4
        batches = [{CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)} for _ in range(acc_steps)]

        model = nn.Linear(4, 1, bias=False)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=batches,
            network=model,
            optimizer=opt,
            loss_function=nn.MSELoss(),
            accumulation_steps=acc_steps,
            decollate=False,
        )
        trainer.run()

        # The output loss should be the full (unscaled) loss value, not divided by acc_steps
        output_loss = trainer.state.output[CommonKeys.LOSS].item()
        self.assertGreater(output_loss, 0.0)

    # ---- accumulation_steps attribute ----

    def test_accumulation_steps_stored(self) -> None:
        """Verify the accumulation_steps attribute is accessible on the trainer."""
        from monai.engines import SupervisedTrainer

        model = nn.Linear(4, 1)
        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=[{CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)}],
            network=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_function=nn.MSELoss(),
            accumulation_steps=8,
        )
        self.assertEqual(trainer.accumulation_steps, 8)

    # ---- default is no accumulation ----

    def test_default_no_accumulation(self) -> None:
        """Default accumulation_steps=1 means no accumulation."""
        from monai.engines import SupervisedTrainer

        model = nn.Linear(4, 1)
        trainer = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=[{CommonKeys.IMAGE: torch.randn(1, 4), CommonKeys.LABEL: torch.randn(1, 1)}],
            network=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_function=nn.MSELoss(),
        )
        self.assertEqual(trainer.accumulation_steps, 1)


if __name__ == "__main__":
    unittest.main()
