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

import os
import tempfile
import unittest
from typing import Dict, List

import nibabel as nib
import numpy as np

from monai.apps.auto3dseg import AutoRunner
from monai.bundle.config_parser import ConfigParser
from monai.data import create_test_image_3d
from monai.utils import optional_import
from tests.utils import SkipIfBeforePyTorchVersion, skip_if_no_cuda, skip_if_quick

_, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")
_, has_nni = optional_import("nni")

sim_datalist: Dict[str, List[Dict]] = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_003.fake.nii.gz", "label": "tr_label_003.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_005.fake.nii.gz", "label": "tr_label_005.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_006.fake.nii.gz", "label": "tr_label_006.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_007.fake.nii.gz", "label": "tr_label_007.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_008.fake.nii.gz", "label": "tr_label_008.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_009.fake.nii.gz", "label": "tr_label_009.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_010.fake.nii.gz", "label": "tr_label_010.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_011.fake.nii.gz", "label": "tr_label_011.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_012.fake.nii.gz", "label": "tr_label_012.fake.nii.gz"},
    ],
}

train_param = {
    "CUDA_VISIBLE_DEVICES": [0],
    "num_iterations": 8,
    "num_iterations_per_validation": 4,
    "num_images_per_batch": 2,
    "num_epochs": 2,
    "num_warmup_iterations": 4,
}

pred_param = {"files_slices": slice(0, 1), "mode": "mean", "sigmoid": True}


@skip_if_quick
@SkipIfBeforePyTorchVersion((1, 9, 1))
@unittest.skipIf(not has_tb, "no tensorboard summary writer")
class TestAutoRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        test_path = self.test_dir.name

        sim_dataroot = os.path.join(test_path, "dataroot")
        if not os.path.isdir(sim_dataroot):
            os.makedirs(sim_dataroot)

        # Generate a fake dataset
        for d in sim_datalist["testing"] + sim_datalist["training"]:
            im, seg = create_test_image_3d(64, 64, 64, rad_max=10, num_seg_classes=1)
            nib_image = nib.Nifti1Image(im, affine=np.eye(4))
            image_fpath = os.path.join(sim_dataroot, d["image"])
            nib.save(nib_image, image_fpath)

            if "label" in d:
                nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
                label_fpath = os.path.join(sim_dataroot, d["label"])
                nib.save(nib_image, label_fpath)

        sim_json_datalist = os.path.join(sim_dataroot, "sim_input.json")
        ConfigParser.export_config_file(sim_datalist, sim_json_datalist)

        data_src_cfg = os.path.join(test_path, "data_src_cfg.yaml")
        data_src = {
            "name": "sim_data",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": sim_json_datalist,
            "dataroot": sim_dataroot,
            "multigpu": False,
            "class_names": ["label_class"],
        }

        ConfigParser.export_config_file(data_src, data_src_cfg)
        self.data_src_cfg = data_src_cfg
        self.test_path = test_path

    @skip_if_no_cuda
    def test_autorunner(self) -> None:
        work_dir = os.path.join(self.test_path, "work_dir")
        runner = AutoRunner(work_dir=work_dir, input=self.data_src_cfg)
        runner.set_training_params(train_param)  # 2 epochs
        runner.set_num_fold(1)
        runner.run()

    @skip_if_no_cuda
    def test_autorunner_hpo(self) -> None:
        if has_nni:
            work_dir = os.path.join(self.test_path, "work_dir")
            runner = AutoRunner(work_dir=work_dir, input=self.data_src_cfg, hpo=True)
            hpo_param = {
                "num_iterations": 8,
                "num_iterations_per_validation": 4,
                "num_images_per_batch": 2,
                "num_epochs": 2,
                "num_warmup_iterations": 4,
                # below are to shorten the time for dints
                "training#num_iterations": 8,
                "training#num_iterations_per_validation": 4,
                "training#num_images_per_batch": 2,
                "training#num_epochs": 2,
                "training#num_warmup_iterations": 4,
                "searching#num_iterations": 8,
                "searching#num_iterations_per_validation": 4,
                "searching#num_images_per_batch": 2,
                "searching#num_epochs": 2,
                "searching#num_warmup_iterations": 4,
            }
            search_space = {"learning_rate": {"_type": "choice", "_value": [0.0001, 0.001, 0.01, 0.1]}}
            runner.set_num_fold(1)
            runner.set_nni_search_space(search_space)
            runner.set_hpo_params(params=hpo_param)
            runner.run()

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
