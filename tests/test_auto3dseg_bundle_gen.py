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

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.apps.auto3dseg import BundleGen, DataAnalyzer
from monai.bundle import ConfigParser
from monai.data import create_test_image_2d, create_test_image_3d
from monai.utils import AlgoRetrieval

sim_datalist_dict = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
    ],
}

TEST_CASES = [
    [{"algo_retrival_methods": AlgoRetrieval.REL, "algo_urls_or_dirs": ""}],
    [{"algo_retrival_methods": AlgoRetrieval.DEV, "algo_urls_or_dirs": AlgoRetrieval.REPO_URL}],
    [
        {
            "algo_retrival_methods": [AlgoRetrieval.REL, AlgoRetrieval.DEV],
            "algo_urls_or_dirs": ["", AlgoRetrieval.REPO_URL],
        }
    ],
]


def create_sim_data(dataroot: str, datalist_dict: dict, sim_dim: tuple, **kwargs) -> None:
    """
    Create simulated data using create_test_image_3d.

    Args:
        dataroot: data directory path that hosts the "nii.gz" image files.
        datalist_dict: a list of data to create.
        sim_dim: the image sizes, for examples: a tuple of (64, 64, 64) for 3d, or (128, 128) for 2d
    """
    if not os.path.isdir(dataroot):
        os.makedirs(dataroot)

    # Generate a fake dataset
    for d in datalist_dict["testing"] + datalist_dict["training"]:
        if len(sim_dim) == 2:  # 2D image
            im, seg = create_test_image_2d(sim_dim[0], sim_dim[1], **kwargs)
        elif len(sim_dim) == 3:  # 3D image
            im, seg = create_test_image_3d(sim_dim[0], sim_dim[1], sim_dim[2], **kwargs)
        elif len(sim_dim) == 4:  # multi-modality 3D image
            im_list = []
            seg_list = []
            for _ in range(sim_dim[3]):
                im_3d, seg_3d = create_test_image_3d(sim_dim[0], sim_dim[1], sim_dim[2], **kwargs)
                im_list.append(im_3d[..., np.newaxis])
                seg_list.append(seg_3d[..., np.newaxis])
            im = np.concatenate(im_list, axis=3)
            seg = np.concatenate(seg_list, axis=3)
        else:
            raise ValueError(f"Invalid argument input. sim_dim has f{len(sim_dim)} values. 2-4 values are expected.")
        nib_image = nib.Nifti1Image(im, affine=np.eye(4))
        image_fpath = os.path.join(dataroot, d["image"])
        nib.save(nib_image, image_fpath)

        if "label" in d:
            nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
            label_fpath = os.path.join(dataroot, d["label"])
            nib.save(nib_image, label_fpath)


class TestBundleGen(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.work_dir = self.test_dir.name
        self.dataroot = os.path.join(self.work_dir, "sim_dataroot")
        self.datalist_file = os.path.join(self.work_dir, "sim_datalist.json")
        self.data_stats_filename = os.path.join(self.work_dir, "data_stats.yaml")
        input_cfg = {"modality": "MRI", "datalist": self.datalist_file, "dataroot": self.dataroot}
        self.input_cfg_filename = os.path.join(self.work_dir, "input.yaml")

        ConfigParser.export_config_file(sim_datalist_dict, self.datalist_file)
        ConfigParser.export_config_file(input_cfg, self.input_cfg_filename)

        sim_dim = (32, 32, 32)
        create_sim_data(
            self.dataroot, sim_datalist_dict, sim_dim, rad_max=max(int(sim_dim[0] / 4), 1), rad_min=1, num_seg_classes=1
        )
        analyser = DataAnalyzer(self.datalist_file, self.dataroot, output_path=self.data_stats_filename)
        analyser.get_all_case_stats()

    @parameterized.expand(TEST_CASES)
    def test_bundle_gen(self, input_params):
        bundle_generator = BundleGen(
            algo_path=self.work_dir,
            algo_retrival_methods=input_params["algo_retrival_methods"],
            algo_urls_or_dirs=input_params["algo_urls_or_dirs"],
            data_stats_filename=self.data_stats_filename,
            data_src_cfg_name=self.input_cfg_filename,
        )
        bundle_generator.generate(output_folder=self.work_dir)

    def tearDown(self):
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
