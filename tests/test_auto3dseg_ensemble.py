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

from monai.apps.auto3dseg import AlgoEnsembleBestByFold, AlgoEnsembleBestN, AlgoEnsembleBuilder, BundleGen, DataAnalyzer
from monai.bundle.config_parser import ConfigParser
from monai.data import create_test_image_3d
from monai.utils.enums import AlgoEnsembleKeys
from tests.utils import skip_if_no_cuda

fake_datalist: Dict[str, List[Dict]] = {
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
}

pred_param = {"files_slices": slice(0, 1), "mode": "mean", "sigmoid": True}


class TestEnsembleBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()

    @skip_if_no_cuda
    def test_ensemble(self) -> None:
        test_path = self.test_dir.name

        dataroot = os.path.join(test_path, "dataroot")
        work_dir = os.path.join(test_path, "workdir")

        da_output_yaml = os.path.join(work_dir, "datastats.yaml")
        data_src_cfg = os.path.join(work_dir, "data_src_cfg.yaml")

        if not os.path.isdir(dataroot):
            os.makedirs(dataroot)

        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        # Generate a fake dataset
        for d in fake_datalist["testing"] + fake_datalist["training"]:
            im, seg = create_test_image_3d(64, 64, 64, rad_max=10, num_seg_classes=1)
            nib_image = nib.Nifti1Image(im, affine=np.eye(4))
            image_fpath = os.path.join(dataroot, d["image"])
            nib.save(nib_image, image_fpath)

            if "label" in d:
                nib_image = nib.Nifti1Image(seg, affine=np.eye(4))
                label_fpath = os.path.join(dataroot, d["label"])
                nib.save(nib_image, label_fpath)

        # write to a json file
        fake_json_datalist = os.path.join(dataroot, "fake_input.json")
        ConfigParser.export_config_file(fake_datalist, fake_json_datalist)

        da = DataAnalyzer(fake_json_datalist, dataroot, output_path=da_output_yaml)
        da.get_all_case_stats()

        data_src = {
            "name": "fake_data",
            "task": "segmentation",
            "modality": "MRI",
            "datalist": fake_json_datalist,
            "dataroot": dataroot,
            "multigpu": False,
            "class_names": ["label_class"],
        }

        ConfigParser.export_config_file(data_src, data_src_cfg)

        bundle_generator = BundleGen(
            algo_path=work_dir, data_stats_filename=da_output_yaml, data_src_cfg_name=data_src_cfg
        )
        bundle_generator.generate(work_dir, num_fold=2)
        history = bundle_generator.get_history()

        for h in history:
            self.assertEqual(len(h.keys()), 1, "each record should have one model")
            for _, algo in h.items():
                algo.train(train_param)

        builder = AlgoEnsembleBuilder(history, data_src_cfg)
        builder.set_ensemble_method(AlgoEnsembleBestN(n_best=2))
        ensemble = builder.get_ensemble()
        preds = ensemble(pred_param)
        self.assertTupleEqual(preds[0].shape, (2, 64, 64, 64))

        builder.set_ensemble_method(AlgoEnsembleBestByFold(2))
        ensemble = builder.get_ensemble()
        for algo in ensemble.get_algo_ensemble():
            print(algo[AlgoEnsembleKeys.ID])

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
