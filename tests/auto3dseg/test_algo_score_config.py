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

import json
import os
import tempfile
import unittest

from monai.apps.auto3dseg.bundle_gen import BundleAlgo
from monai.apps.auto3dseg.utils import import_bundle_algo_history


class TestAlgoScoreConfig(unittest.TestCase):
    """Regression tests for the score of an algorithm output directory.

    ``import_bundle_algo_history`` reads any work directory, so the directory and the config inside it
    can come from someone else. Reading the saved score must not evaluate code from that config.
    """

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = self.tempdir.name
        self.marker = os.path.join(self.root, "PWNED")
        self.algo_dir = os.path.join(self.root, "algo1")
        os.makedirs(os.path.join(self.algo_dir, "configs"))

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_config(self, ckpt_value: str) -> None:
        with open(os.path.join(self.algo_dir, "configs", "hyper_parameters.yaml"), "w") as f:
            f.write(f"ckpt_path: {json.dumps(ckpt_value)}\n")

    def _write_progress(self, ckpt_dir: str, score: float) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "progress.yaml"), "w") as f:
            f.write(f"- best_avg_dice_score: {score}\n")

    def _algo(self) -> BundleAlgo:
        algo = BundleAlgo(template_path=self.root)
        algo.output_path = self.algo_dir
        return algo

    def test_expression_in_ckpt_path_does_not_run(self):
        """A ``$`` expression in ``ckpt_path`` cannot import or call anything."""
        payload = f"$__import__('pathlib').Path({self.marker!r}).write_text('pwned') or {self.algo_dir!r}"
        self._write_config(payload)

        with self.assertRaises(RuntimeError):
            self._algo().get_score()
        self.assertFalse(os.path.exists(self.marker), "the ckpt_path expression was evaluated")

    def test_module_names_are_not_in_scope(self):
        """The evaluation namespace has no monai/torch/numpy, even though the default parser provides them."""
        self._write_config("$str(torch.tensor(1))")

        with self.assertRaises(RuntimeError):
            self._algo().get_score()
        self.assertFalse(os.path.exists(self.marker))

    def test_string_expression_still_resolves(self):
        """The documented string form, for example ``$@bundle_root + '/model_fold_0'``, keeps working."""
        ckpt_dir = os.path.join(self.algo_dir, "model_fold_0")
        self._write_progress(ckpt_dir, 0.42)
        self._write_config(f"${self.algo_dir!r} + '/model_fold_0'")

        self.assertAlmostEqual(self._algo().get_score(), 0.42)

    def test_plain_ckpt_path_still_resolves(self):
        ckpt_dir = os.path.join(self.algo_dir, "ckpt")
        self._write_progress(ckpt_dir, 0.75)
        self._write_config(ckpt_dir)

        self.assertAlmostEqual(self._algo().get_score(), 0.75)

    def test_history_import_does_not_run_a_config_payload(self):
        """End to end: a work directory with a malicious config is imported without executing it."""
        payload = f"$__import__('pathlib').Path({self.marker!r}).write_text('pwned') or {self.algo_dir!r}"
        self._write_config(payload)
        with open(os.path.join(self.algo_dir, "algo_object.json"), "w") as f:
            json.dump(
                {
                    "_target_": "monai.apps.auto3dseg.bundle_gen.BundleAlgo",
                    "_state_": {"template_path": self.root, "output_path": self.algo_dir},
                },
                f,
            )

        with self.assertLogs(level="WARNING"):
            history = import_bundle_algo_history(self.root, only_trained=False)

        self.assertFalse(os.path.exists(self.marker), "the work directory payload was evaluated")
        self.assertEqual(len(history), 1)
        self.assertFalse(history[0]["is_trained"])


if __name__ == "__main__":
    unittest.main()
