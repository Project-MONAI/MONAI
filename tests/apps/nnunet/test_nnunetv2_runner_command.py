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

import os
import sys
import types
import unittest
import warnings
from unittest import mock

from monai.apps.nnunet import nnunetv2_runner
from monai.apps.nnunet.nnunetv2_runner import nnUNetV2Runner
from monai.bundle import ConfigParser


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


class TestTrainParallelCommand(unittest.TestCase):
    def test_train_parallel_uses_argv_list_without_shell(self):
        runner = _make_runner()
        runner.dataset_name = "Dataset001_Test"
        runner.nnunet_results = "/tmp/nnunet_results"

        all_cmds = [
            {
                0: [
                    (["python", "-m", "train", "--fold", "0"], {"CUDA_VISIBLE_DEVICES": "0"}),
                    (["python", "-m", "train", "--fold", "1"], {"CUDA_VISIBLE_DEVICES": "0"}),
                ],
                1: [(["python", "-m", "train", "--fold", "2"], {"CUDA_VISIBLE_DEVICES": "1"})],
            }
        ]

        with mock.patch.object(runner, "train_parallel_cmd", return_value=all_cmds):
            with mock.patch("monai.apps.nnunet.nnunetv2_runner.subprocess.Popen") as popen:
                popen.return_value.wait.return_value = None
                runner.train_parallel()

        self.assertEqual(popen.call_count, 3)
        for call in popen.call_args_list:
            self.assertIsInstance(call.args[0], list)
            self.assertFalse(call.kwargs["shell"])


class TestPredictEnsemblePostprocessingWarnings(unittest.TestCase):
    def test_postprocessing_pickle_warns_on_untrusted_file(self):
        runner = _make_runner()
        runner.dataset_name = "Dataset001_Test"
        runner.nnunet_raw = "/tmp/nnunet_raw"
        runner.nnunet_results = "/tmp/nnunet_results"
        runner.best_configuration = {
            "best_model_or_ensemble": {
                "selected_model_or_models": [{"configuration": "3d_fullres"}],
                "postprocessing_file": "/tmp/attacker_controlled_postprocessing.pkl",
                "some_plans_file": "/tmp/plans.json",
            }
        }

        ensemble_mod = types.ModuleType("nnunetv2.ensembling.ensemble")
        ensemble_mod.ensemble_folders = mock.MagicMock()
        pp_mod = types.ModuleType("nnunetv2.postprocessing.remove_connected_components")
        pp_mod.apply_postprocessing_to_folder = mock.MagicMock()
        fp_mod = types.ModuleType("nnunetv2.utilities.file_path_utilities")
        fp_mod.get_output_folder = mock.MagicMock(return_value="/tmp/model_folder")

        fake_modules = {
            "nnunetv2.ensembling.ensemble": ensemble_mod,
            "nnunetv2.postprocessing.remove_connected_components": pp_mod,
            "nnunetv2.utilities.file_path_utilities": fp_mod,
        }

        load_pickle = mock.MagicMock(return_value=([], {}))
        with mock.patch.dict(sys.modules, fake_modules):
            with mock.patch.object(ConfigParser, "load_config_file", return_value=runner.best_configuration):
                with mock.patch.object(nnunetv2_runner, "join", os.path.join):
                    with mock.patch.object(nnunetv2_runner, "load_pickle", load_pickle):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            runner.predict_ensemble_postprocessing(
                                run_predict=False, run_ensemble=False, run_postprocessing=True
                            )

        load_pickle.assert_called_once_with("/tmp/attacker_controlled_postprocessing.pkl")
        self.assertTrue(any("unpickling postprocessing_file" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
