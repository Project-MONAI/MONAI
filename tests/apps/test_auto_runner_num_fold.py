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
import logging
import tempfile
import unittest
from pathlib import Path

from parameterized import parameterized

from monai.apps.auto3dseg import AutoRunner
from monai.utils import optional_import

_, has_sklearn = optional_import("sklearn.model_selection", name="KFold")
_, has_yaml = optional_import("yaml")


@unittest.skipUnless(has_sklearn and has_yaml, "scikit-learn and PyYAML required")
class TestAutoRunnerNumFold(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.tmp_path = Path(temp_dir.name)

    def test_autorunner_generates_configured_num_fold(self):
        tmp_path = self.tmp_path
        datalist_path = tmp_path / "datalist.json"
        datalist_path.write_text(
            json.dumps({"training": [{"image": f"image_{i}.nii.gz", "label": f"label_{i}.nii.gz"} for i in range(10)]}),
            encoding="utf-8",
        )
        runner = AutoRunner(
            work_dir=str(tmp_path / "work"),
            input={"modality": "CT", "dataroot": str(tmp_path), "datalist": str(datalist_path), "num_fold": 2},
            analyze=False,
            algo_gen=False,
            train=False,
            ensemble=False,
        )
        with open(runner.datalist_filename, encoding="utf-8") as f:
            generated = json.load(f)

        assert runner.num_fold == 2
        assert {item["fold"] for item in generated["training"]} == {0, 1}

    @parameterized.expand([("default",), ("existing_folds",), ("validation",), ("six_folds",)])
    def test_autorunner_fold_compatibility(self, case):
        tmp_path = self.tmp_path
        training = [{"image": f"image_{i}.nii.gz", "label": f"label_{i}.nii.gz"} for i in range(10)]
        datalist = {"training": training}
        datalist_path = tmp_path / "datalist.json"
        config = {"modality": "CT", "dataroot": str(tmp_path), "datalist": str(datalist_path)}
        expected_training = None
        expected_num_fold = 5

        if case == "existing_folds":
            for i, item in enumerate(training):
                item["fold"] = i % 5
            config["num_fold"] = expected_num_fold = 2
            expected_training = training
        elif case == "validation":
            # Avoid an existing malformed INFO message in the validation merge path.
            logger = logging.getLogger("monai.apps.auto3dseg.auto_runner")
            self.addCleanup(logger.setLevel, logger.level)
            logger.setLevel(logging.WARNING)
            # Include an overlapping case and a validation-only case to check merging.
            datalist["validation"] = [training[0].copy(), {"image": "val.nii.gz", "label": "val_label.nii.gz"}]
            config["num_fold"] = expected_num_fold = 1
            expected_training = [dict(item, fold=0 if i == 0 else 1) for i, item in enumerate(training)]
            expected_training.append(dict(datalist["validation"][1], fold=0))
        elif case == "six_folds":
            config["num_fold"] = expected_num_fold = 6

        datalist_path.write_text(json.dumps(datalist), encoding="utf-8")
        runner = AutoRunner(
            work_dir=str(tmp_path / "work"), input=config, analyze=False, algo_gen=False, train=False, ensemble=False
        )
        with open(runner.datalist_filename, encoding="utf-8") as f:
            generated = json.load(f)

        assert runner.num_fold == expected_num_fold
        if expected_training is not None:
            assert generated["training"] == expected_training
        else:
            assert len(generated["training"]) == 10
            assert {item["fold"] for item in generated["training"]} == set(range(expected_num_fold))
        assert json.loads(datalist_path.read_text(encoding="utf-8")) == datalist
