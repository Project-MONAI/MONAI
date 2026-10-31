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
import shlex
import unittest
from unittest import mock

from monai.apps.auto3dseg.bundle_gen import BundleAlgo
from monai.apps.auto3dseg.ensemble_builder import EnsembleRunner
from monai.auto3dseg.utils import _prepare_cmd_default, _run_cmd_torchrun, check_and_set_optional_args


def _device_setting(n_devices: int = 1) -> dict:
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(n_devices)),
        "n_devices": n_devices,
        "NUM_NODES": 1,
        "MN_START_METHOD": "bcprun",
        "CMD_PREFIX": "",
    }


class TestOptionalArgs(unittest.TestCase):
    def test_plain_values_keep_their_tokens(self):
        """Values without special characters produce the same arguments as before."""
        cmd = _prepare_cmd_default("train.py run", num_epochs=2, lr=0.1, config_file=["a.yaml", "b.yaml"])
        self.assertEqual(
            shlex.split(cmd),
            ["python", "train.py", "run", "--num_epochs=2", "--lr=0.1", "--config_file='a.yaml,b.yaml'"],
        )

    def test_value_with_whitespace_is_one_argument(self):
        for value in ("/data/my dir", "/tmp/x\t--work_dir\t/other", "a b --c=d"):
            with self.subTest(value=value):
                args = shlex.split(check_and_set_optional_args({"work_dir": value}))
                self.assertEqual(args, [f"--work_dir={value}"])

    def test_torchrun_keeps_argument_boundaries(self):
        cmd = "train.py run" + check_and_set_optional_args({"output_dir": "/tmp/a b"})
        with mock.patch("monai.auto3dseg.utils.run_cmd") as run:
            _run_cmd_torchrun(cmd, nnodes=1, nproc_per_node=2)
        self.assertEqual(
            run.call_args.args[0],
            ["torchrun", "--nnodes", "1", "--nproc_per_node", "2", "train.py", "run", "--output_dir=/tmp/a b"],
        )


class TestEnsembleRunnerCommand(unittest.TestCase):
    def _runner(self, work_dir: str, **kwargs) -> EnsembleRunner:
        runner = EnsembleRunner.__new__(EnsembleRunner)
        runner.data_src_cfg_name = "input.yaml"
        runner.work_dir = work_dir
        runner.num_fold = 5
        runner.ensemble_method_name = "AlgoEnsembleBestByFold"
        runner.kwargs = kwargs
        runner.device_setting = _device_setting(n_devices=2)
        return runner

    def test_values_stay_single_arguments(self):
        work_dir = "/tmp/my work\t--work_dir\t/elsewhere"
        runner = self._runner(work_dir, sigmoid="True --output_dir=/elsewhere")
        with mock.patch("monai.apps.auto3dseg.ensemble_builder._run_cmd_torchrun") as run:
            runner._create_cmd()
        args = shlex.split(run.call_args.args[0])
        self.assertEqual(args[args.index("--work_dir") + 1], work_dir)
        self.assertEqual(args.count("--work_dir"), 1)
        self.assertIn("--sigmoid=True --output_dir=/elsewhere", args)
        self.assertFalse(any(a.startswith("--output_dir") for a in args))


class TestBundleAlgoCommand(unittest.TestCase):
    def test_output_path_with_whitespace_is_one_argument(self):
        algo = BundleAlgo(template_path=None)
        algo.output_path = "/tmp/my algo"
        algo.device_setting = _device_setting(n_devices=1)
        with mock.patch("monai.apps.auto3dseg.bundle_gen.run_cmd") as run:
            algo.train({"num_epochs": 1})
        args = run.call_args.args[0]
        self.assertEqual(args[:3], ["python", os.path.join("/tmp/my algo", "scripts", "train.py"), "run"])
        self.assertIn("--num_epochs=1", args)


class TestBundleAlgoLoadState(unittest.TestCase):
    def test_round_trip(self):
        algo = BundleAlgo(template_path="/templates")
        algo.name = "segresnet_0"
        algo.device_setting = _device_setting(n_devices=2)
        state = algo.state_dict()
        restored = BundleAlgo()
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)

    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected keys"):
            BundleAlgo().load_state_dict({"train": "x"})

    def test_malformed_device_setting_is_rejected(self):
        for setting in (
            {"n_devices": "2 --master_port 1"},
            {"NUM_NODES": "1 2"},
            {"CUDA_VISIBLE_DEVICES": "0 --x"},
            {"CMD_PREFIX": ["python"]},
            {"OTHER": 1},
            "not a dict",
        ):
            with self.subTest(setting=setting):
                with self.assertRaises(ValueError):
                    BundleAlgo().load_state_dict({"device_setting": setting})

    def test_partial_device_setting_keeps_defaults(self):
        algo = BundleAlgo()
        defaults = dict(algo.device_setting)
        algo.load_state_dict({"device_setting": {"CUDA_VISIBLE_DEVICES": "0", "n_devices": 1, "NUM_NODES": 1}})
        self.assertEqual(algo.device_setting["CMD_PREFIX"], defaults["CMD_PREFIX"])
        self.assertEqual(algo.device_setting["MN_START_METHOD"], defaults["MN_START_METHOD"])
        self.assertEqual(algo.device_setting["CUDA_VISIBLE_DEVICES"], "0")

    def test_numeric_strings_are_converted(self):
        algo = BundleAlgo()
        algo.load_state_dict({"device_setting": {"n_devices": "2", "NUM_NODES": "1", "CUDA_VISIBLE_DEVICES": "0,1"}})
        self.assertEqual(algo.device_setting["n_devices"], 2)
        self.assertEqual(algo.device_setting["NUM_NODES"], 1)

    def test_missing_n_devices_follows_visible_devices(self):
        """A state that only names the visible devices must not keep the device count of the current host."""
        algo = BundleAlgo()
        algo.device_setting = _device_setting(n_devices=4)
        algo.load_state_dict({"device_setting": {"CUDA_VISIBLE_DEVICES": "0"}})
        self.assertEqual(algo.device_setting["n_devices"], 1)
        self.assertEqual(algo.device_setting["CUDA_VISIBLE_DEVICES"], "0")

    def test_n_devices_above_visible_devices_is_rejected(self):
        algo = BundleAlgo()
        algo.device_setting = _device_setting(n_devices=4)
        with self.assertRaisesRegex(ValueError, "exposes 1"):
            algo.load_state_dict({"device_setting": {"CUDA_VISIBLE_DEVICES": "0", "n_devices": 4}})

    def test_mig_device_identifiers_are_accepted(self):
        mig = "MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2"
        algo = BundleAlgo()
        algo.load_state_dict({"device_setting": {"CUDA_VISIBLE_DEVICES": mig}})
        self.assertEqual(algo.device_setting["CUDA_VISIBLE_DEVICES"], mig)
        self.assertEqual(algo.device_setting["n_devices"], 1)

    def test_state_keys_do_not_need_a_constructed_instance(self):
        """A deserializer that builds the object without ``__init__`` still gets key validation, not AttributeError."""
        algo = object.__new__(BundleAlgo)
        algo.load_state_dict({"template_path": "$len([1, 2])"})
        self.assertEqual(algo.template_path, "$len([1, 2])")


if __name__ == "__main__":
    unittest.main()
