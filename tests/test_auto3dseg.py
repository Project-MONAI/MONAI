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

import sys
import tempfile
import unittest
from copy import deepcopy
from os import path

import nibabel as nib
import numpy as np
import torch

from monai import data
from monai.auto3dseg.analyzer import Analyzer
from monai.auto3dseg.data_analyzer import DataAnalyzer
from monai.auto3dseg.operations import Operations
from monai.auto3dseg.utils import datafold_read
from monai.bundle import ConfigParser
from monai.data import create_test_image_3d
from monai.data.utils import no_collation
from monai.transforms import Compose, LoadImaged

device = "cuda" if torch.cuda.is_available() else "cpu"
n_workers = 0 if sys.platform in ("win32", "darwin") else 2

fake_datalist = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
    ],
}


class TestOperations(Operations):
    """
    Test example for user operation
    """

    def __init__(self) -> None:
        self.data = {"max": np.max, "mean": np.mean, "min": np.min}


class TestAnalyzer(Analyzer):
    """
    Test example for a simple Analyzer
    """

    def __init__(self, report_format):
        super().__init__(report_format)

    def __call__(self, data):
        report = deepcopy(self.get_report_format())
        report["stats"] = self.ops["stats"].evaluate(data)
        return report


class TestImageAnalyzer(Analyzer):
    """
    Test example for a simple Analyzer
    """

    def __init__(self, image_key="image"):

        self.image_key = image_key
        report_format = {"stats": None}

        super().__init__(report_format)
        self.update_ops("stats", TestOperations())

    def __call__(self, data):
        nda = data[self.image_key]
        report = deepcopy(self.get_report_format())
        report["stats"] = self.ops["stats"].evaluate(nda)
        return report


class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        dataroot = self.test_dir.name

        # Generate a fake dataset
        for d in fake_datalist["testing"] + fake_datalist["training"]:
            im, seg = create_test_image_3d(39, 47, 46, rad_max=10)
            nib_image = nib.Nifti1Image(im, affine=np.eye(4))
            image_fpath = path.join(dataroot, d["image"])
            nib.save(nib_image, image_fpath)

            if "label" in d:
                nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
                label_fpath = path.join(dataroot, d["label"])
                nib.save(nib_image, label_fpath)

        # write to a json file
        self.fake_json_datalist = path.join(dataroot, "fake_input.json")
        ConfigParser.export_config_file(fake_datalist, self.fake_json_datalist)

    def test_data_analyzer(self):
        dataroot = self.test_dir.name
        yaml_fpath = path.join(dataroot, "data_stats.yaml")
        analyser = DataAnalyzer(fake_datalist, dataroot, output_path=yaml_fpath, device=device, worker=n_workers)
        datastat = analyser.get_all_case_stats()

        assert len(datastat["stats_by_cases"]) == len(fake_datalist["training"])

    def test_data_analyzer_image_only(self):
        dataroot = self.test_dir.name
        yaml_fpath = path.join(dataroot, "data_stats.yaml")
        analyser = DataAnalyzer(
            fake_datalist, dataroot, output_path=yaml_fpath, device=device, worker=n_workers, label_key=None
        )
        datastat = analyser.get_all_case_stats()

        assert len(datastat["stats_by_cases"]) == len(fake_datalist["training"])

    def test_data_analyzer_from_yaml(self):
        dataroot = self.test_dir.name
        yaml_fpath = path.join(dataroot, "data_stats.yaml")
        analyser = DataAnalyzer(
            self.fake_json_datalist, dataroot, output_path=yaml_fpath, device=device, worker=n_workers
        )
        datastat = analyser.get_all_case_stats()

        assert len(datastat["stats_by_cases"]) == len(fake_datalist["training"])

    def test_basic_analyzer_class(self):
        test_data = np.random.rand(10, 10)
        report_format = {"stats": None}
        user_analyzer = TestAnalyzer(report_format)
        user_analyzer.update_ops("stats", TestOperations())
        result = user_analyzer(test_data)
        assert result["stats"]["max"] == np.max(test_data)
        assert result["stats"]["min"] == np.min(test_data)
        assert result["stats"]["mean"] == np.mean(test_data)

    def test_transform_analyzer_class(self):
        transform_list = [LoadImaged(keys=["image"]), TestImageAnalyzer(image_key="image")]
        transform = Compose(transform_list)
        dataroot = self.test_dir.name
        files, _ = datafold_read(self.fake_json_datalist, dataroot, fold=-1)
        ds = data.Dataset(data=files)
        self.dataset = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=no_collation)
        for batch_data in self.dataset:
            result = transform(batch_data[0])
            assert "stats" in result
            assert "max" in result["stats"]
            assert "min" in result["stats"]
            assert "mean" in result["stats"]

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
