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

    def _write_config(self, **values) -> None:
        with open(os.path.join(self.algo_dir, "configs", "hyper_parameters.yaml"), "w") as f:
            for key, value in values.items():
                f.write(f"{key}: {json.dumps(value)}\n")

    def _write_progress(self, ckpt_dir: str, score: float) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "progress.yaml"), "w") as f:
            f.write(f"- best_avg_dice_score: {score}\n")

    def _algo(self) -> BundleAlgo:
        algo = BundleAlgo(template_path=self.root)
        algo.output_path = self.algo_dir
        return algo

    def _assert_refused(self, **values) -> None:
        self._write_config(**values)
        with self.assertRaises(ValueError):
            self._algo().get_score()
        self.assertFalse(os.path.exists(self.marker), "the ckpt_path payload was executed")

    def test_plain_ckpt_path_still_resolves(self):
        ckpt_dir = os.path.join(self.algo_dir, "ckpt")
        self._write_progress(ckpt_dir, 0.75)
        self._write_config(ckpt_path=ckpt_dir)

        self.assertAlmostEqual(self._algo().get_score(), 0.75)

    def test_string_concatenation_still_resolves(self):
        ckpt_dir = os.path.join(self.algo_dir, "model_fold_0")
        self._write_progress(ckpt_dir, 0.42)
        self._write_config(ckpt_path=f"${self.algo_dir!r} + '/model_fold_0'")

        self.assertAlmostEqual(self._algo().get_score(), 0.42)

    def test_reference_expression_still_resolves(self):
        """The form used by the shipped templates, ``$@bundle_root + '/model'``, keeps working."""
        ckpt_dir = os.path.join(self.algo_dir, "model")
        self._write_progress(ckpt_dir, 0.33)
        self._write_config(bundle_root=self.algo_dir, ckpt_path="$@bundle_root + '/model'")

        self.assertAlmostEqual(self._algo().get_score(), 0.33)

    def test_integer_reference_through_str_still_resolves(self):
        """The ``dints`` and ``swinunetr`` templates use ``$@bundle_root + '/model_fold' + str(@fold)``."""
        ckpt_dir = os.path.join(self.algo_dir, "model_fold0")
        self._write_progress(ckpt_dir, 0.61)
        self._write_config(bundle_root=self.algo_dir, fold=0, ckpt_path="$@bundle_root + '/model_fold' + str(@fold)")

        self.assertAlmostEqual(self._algo().get_score(), 0.61)

    def test_integer_reference_cannot_be_concatenated(self):
        """An integer reference is only usable through str(); concatenating it directly is refused."""
        self._assert_refused(bundle_root=self.algo_dir, fold=0, ckpt_path="$@bundle_root + @fold")

    def test_missing_reference_is_refused(self):
        """A reference to an id that is not in the file raises ValueError with the documented message."""
        self._write_config(ckpt_path="$@no_such_entry + '/model'")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self._algo().get_score()

    def test_unknown_reference_key_is_refused(self):
        """A subscript that does not come from a matched @reference raises ValueError, not KeyError."""
        self._assert_refused(ckpt_path="$__local_refs['missing']")

    def test_whole_value_reference_still_resolves(self):
        ckpt_dir = os.path.join(self.algo_dir, "ckpt")
        self._write_progress(ckpt_dir, 0.5)
        self._write_config(ckpt=ckpt_dir, ckpt_path="@ckpt")

        self.assertAlmostEqual(self._algo().get_score(), 0.5)

    def test_expression_in_ckpt_path_does_not_run(self):
        """A ``$`` expression in ``ckpt_path`` cannot import or call anything."""
        payload = f"$__import__('pathlib').Path({self.marker!r}).write_text('pwned') or {self.algo_dir!r}"

        self._assert_refused(ckpt_path=payload)

    def test_module_names_are_not_in_scope(self):
        self._assert_refused(ckpt_path="$str(torch.tensor(1))")

    def test_subclasses_escape_is_refused(self):
        """The usual ``__subclasses__()`` sandbox escape has no way in: no attribute or call is accepted."""
        payload = (
            f"$[c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'catch_warnings'][0]()"
            f"._module.__builtins__['__import__']('pathlib').Path({self.marker!r}).write_text('pwned')"
        )

        self._assert_refused(ckpt_path=payload)

    def test_target_mapping_is_refused(self):
        """A ``_target_`` mapping is not instantiated: ``get_score`` never asks for a component."""
        self._assert_refused(ckpt_path={"_target_": "subprocess.Popen", "args": [["touch", self.marker]]})

    def test_history_import_does_not_run_a_config_payload(self):
        """End to end: a work directory with a malicious config is imported without executing it."""
        payload = f"$__import__('pathlib').Path({self.marker!r}).write_text('pwned') or {self.algo_dir!r}"
        self._write_config(ckpt_path=payload)
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
