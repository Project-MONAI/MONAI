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
from os import path

import nibabel as nib
import numpy as np
import torch

from monai.apps.auto3d.data_analyzer import DataAnalyzer
from monai.apps.auto3d.algorithm_autoconfig import auto_configer
from monai.apps.auto3d.distributed_trainer import DistributedTrainer
from monai.data import create_test_image_3d

device = "cuda" if torch.cuda.is_available() else "cpu"
n_workers = 0 if sys.platform in ("win32", "darwin") else 2

fake_datalist = {
    "testing": [{"image": "val_001.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_003.fake.nii.gz", "label": "tr_label_003.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_005.fake.nii.gz", "label": "tr_label_005.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_006.fake.nii.gz", "label": "tr_label_006.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_007.fake.nii.gz", "label": "tr_label_007.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_008.fake.nii.gz", "label": "tr_label_008.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_005.fake.nii.gz", "label": "tr_label_005.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_006.fake.nii.gz", "label": "tr_label_006.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_007.fake.nii.gz", "label": "tr_label_007.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_008.fake.nii.gz", "label": "tr_label_008.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_009.fake.nii.gz", "label": "tr_label_009.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_010.fake.nii.gz", "label": "tr_label_010.fake.nii.gz"},
    ],
}


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

    def test_data_analyzer(self):
        dataroot = self.test_dir.name
        yaml_fpath = path.join(dataroot, "data_stats.yaml")
        analyser = DataAnalyzer(fake_datalist, dataroot, output_path=yaml_fpath, device=device, worker=n_workers)
        datastat = analyser.get_all_case_stats()
        print(len(datastat["stats_by_cases"]))
        assert len(datastat["stats_by_cases"]) == len(fake_datalist["training"])

    def test_algo_autoconfig(self):
        dataroot = self.test_dir.name
        script_dir = path.join(self.test_dir.name, "scripts")
        yaml_fpath = path.join(dataroot, "data_stats.yaml")
        analyser = DataAnalyzer(fake_datalist, dataroot, output_path=yaml_fpath, device=device, worker=n_workers)
        datastat = analyser.get_all_case_stats()
        input_args = {
            "datastat": datastat,
            "datalist": fake_datalist,
            "dataroot": dataroot,
            "name": "UnitTest",
            "task": "segmentation",
            "modality": "MRI",
            "multigpu": True,
        }

        networks = ["UNet"]
        for net in networks:
            configer = auto_configer(net, **input_args)
            configer.generate_scripts(script_dir)

    def test_distributed_train(self):
        dataroot = self.test_dir.name
        script_dir = path.join(self.test_dir.name, "scripts")
        yaml_fpath = path.join(self.test_dir.name, "data_stats.yaml")
        analyser = DataAnalyzer(fake_datalist, dataroot, output_path=yaml_fpath, device=device, worker=n_workers)
        datastat = analyser.get_all_case_stats()
        input_args = {
            "datastat": datastat,
            "datalist": fake_datalist,
            "dataroot": dataroot,
            "name": "UnitTest",
            "task": "segmentation",
            "modality": "MRI",
            "multigpu": False,
        }

        networks = ["UNet"]
        for net in networks:
            configer = auto_configer(net, **input_args)
            config = configer.generate_scripts(script_dir)

        model = DistributedTrainer(config)
        model.train()

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
