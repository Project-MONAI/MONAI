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
from unittest import mock

from monai.apps.nnunet.nnunetv2_runner import nnUNetV2Runner


def _make_runner(export_validation_probabilities=False):
    runner = nnUNetV2Runner.__new__(nnUNetV2Runner)
    runner.dataset_name_or_id = "001"
    runner.trainer_class_name = "nnUNetTrainer"
    runner.export_validation_probabilities = export_validation_probabilities
    return runner


class TestTrainSingleModelCommand(unittest.TestCase):
    def test_store_true_flags_emit_bare(self):
        runner = _make_runner()
        cmd, _ = runner.train_single_model_command(
            "3d_fullres", 0, 0, {"c": True, "val": True, "use_compressed": True, "disable_checkpointing": True}
        )
        for flag in ("--c", "--val", "--use_compressed", "--disable_checkpointing"):
            self.assertIn(flag, cmd)
        self.assertNotIn("True", cmd)

    def test_store_true_flags_false_omitted(self):
        runner = _make_runner()
        cmd, _ = runner.train_single_model_command(
            "3d_fullres", 0, 0, {"c": False, "val": False, "use_compressed": False, "disable_checkpointing": False}
        )
        for flag in ("--c", "--val", "--use_compressed", "--disable_checkpointing"):
            self.assertNotIn(flag, cmd)
        self.assertNotIn("False", cmd)

    def test_pretrained_weights_truthy_included(self):
        runner = _make_runner()
        cmd, _ = runner.train_single_model_command("3d_fullres", 0, 0, {"pretrained_weights": "/path/to/weights.pth"})
        self.assertIn("-pretrained_weights", cmd)
        self.assertIn("/path/to/weights.pth", cmd)

    def test_pretrained_weights_falsy_omitted(self):
        runner = _make_runner()
        cmd, _ = runner.train_single_model_command("3d_fullres", 0, 0, {"pretrained_weights": False})
        self.assertNotIn("-pretrained_weights", cmd)
        self.assertNotIn("False", cmd)

    def test_value_kwargs_unaffected(self):
        runner = _make_runner()
        cmd, _ = runner.train_single_model_command("3d_fullres", 0, 0, {"npz": "something"})
        self.assertIn("--npz", cmd)
        self.assertIn("something", cmd)


class TestValidateSingleModelCommand(unittest.TestCase):
    def test_validate_emits_bare_val_flag(self):
        runner = _make_runner()
        with mock.patch("monai.apps.nnunet.nnunetv2_runner.run_cmd") as run_cmd:
            runner.validate_single_model("3d_fullres", 0)
        cmd = run_cmd.call_args.args[0]
        self.assertIn("--val", cmd)
        self.assertNotIn("--only_run_validation", cmd)
        self.assertNotIn("True", cmd)


if __name__ == "__main__":
    unittest.main()
