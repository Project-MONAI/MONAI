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

import unittest
from os import makedirs, path, remove, rmdir
import sys

import torch
import nibabel as nib
import numpy as np

from monai.apps.auto3d.data_analyzer import DataAnalyzer
from monai.data import create_test_image_3d

device = "cuda" if torch.cuda.is_available() else "cpu"
n_workers = 0 if sys.platform in ("win32", "darwin") else 2

class TestDataAnalyzer(unittest.TestCase):
    def test_data_analyzer(self):
        source_datalist = dict(
            [
                ("testing", [{"image": "val_001.fake.nii.gz"}]),
                (
                    "training",
                    [
                        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
                        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
                        {"fold": 1, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
                        {"fold": 1, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
                    ],
                ),
            ]
        )

        # Generate datasets
        tmp_dir = "tests/testing_data/auto3d_tmp"
        analyzer_output = "output.yaml"

        # generate fake datasets in temporary directory

        if not path.isdir(tmp_dir):
            makedirs(tmp_dir)

        cleanup_list = list()
        for d in source_datalist["testing"] + source_datalist["training"]:
            im, seg = create_test_image_3d(39, 47, 46, rad_max=10)
            nib_image = nib.Nifti1Image(im, affine=np.eye(4))
            image_fpath = path.join(tmp_dir, d["image"])
            nib.save(nib_image, image_fpath)
            cleanup_list.append(image_fpath)
            if "label" in d:
                nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
                label_fpath = path.join(tmp_dir, d["label"])
                nib.save(nib_image, label_fpath)
                cleanup_list.append(label_fpath)

        yaml_fpath = path.join(tmp_dir, analyzer_output)
        analyser = DataAnalyzer(source_datalist, tmp_dir, output_yaml=yaml_fpath, device=device, worker=n_workers)
        analyser_results = analyser.get_all_case_stats()
        cleanup_list.append(yaml_fpath)

        assert len(analyser_results["stats_by_cases"]) == 4

        # clean up the fake datasets and output
        for file in cleanup_list:
            if path.isfile(file):
                remove(file)

        rmdir(tmp_dir)


if __name__ == "__main__":
    unittest.main()
