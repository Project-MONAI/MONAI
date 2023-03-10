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
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from monai.apps.auto3dseg import AlgoEnsembleBestByFold, AlgoEnsembleBestN, AlgoEnsembleBuilder, BundleGen, DataAnalyzer
from monai.bundle.config_parser import ConfigParser
from monai.data import create_test_image_3d
from monai.utils import optional_import, set_determinism
from monai.utils.enums import AlgoEnsembleKeys
from tests.utils import (
    SkipIfBeforePyTorchVersion,
    get_testing_algo_template_path,
    skip_if_downloading_fails,
    skip_if_no_cuda,
    skip_if_quick,
)

_, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")

fake_datalist: dict[str, list[dict]] = {
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

train_param = (
    {
        "num_images_per_batch": 2,
        "num_epochs": 2,
        "num_epochs_per_validation": 1,
        "num_warmup_epochs": 1,
        "use_pretrain": False,
        "pretrained_path": "",
        "determ": True,
    }
    if torch.cuda.is_available()
    else {}
)

pred_param = {"files_slices": slice(0, 1), "mode": "mean", "sigmoid": True}


@skip_if_quick
@SkipIfBeforePyTorchVersion((1, 10, 0))
@unittest.skipIf(not has_tb, "no tensorboard summary writer")
class TestEnsembleBuilder(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(0)
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
            im, seg = create_test_image_3d(24, 24, 24, rad_max=10, num_seg_classes=1)
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

        with skip_if_downloading_fails():
            bundle_generator = BundleGen(
                algo_path=work_dir,
                data_stats_filename=da_output_yaml,
                data_src_cfg_name=data_src_cfg,
                templates_path_or_url=get_testing_algo_template_path(),
            )
        bundle_generator.generate(work_dir, num_fold=1)
        history = bundle_generator.get_history()

        for h in history:
            self.assertEqual(len(h.keys()), 1, "each record should have one model")
            for name, algo in h.items():
                _train_param = train_param.copy()
                if name.startswith("segresnet"):
                    _train_param["network#init_filters"] = 8
                    _train_param["pretrained_ckpt_name"] = ""
                elif name.startswith("swinunetr"):
                    _train_param["network#feature_size"] = 12
                algo.train(_train_param)

        builder = AlgoEnsembleBuilder(history, data_src_cfg)
        builder.set_ensemble_method(AlgoEnsembleBestN(n_best=1))
        ensemble = builder.get_ensemble()
        name = ensemble.get_algo_ensemble()[0][AlgoEnsembleKeys.ID]
        if name.startswith("segresnet"):
            pred_param["network#init_filters"] = 8
        elif name.startswith("swinunetr"):
            pred_param["network#feature_size"] = 12
        preds = ensemble(pred_param)
        self.assertTupleEqual(preds[0].shape, (2, 24, 24, 24))

        builder.set_ensemble_method(AlgoEnsembleBestByFold(1))
        ensemble = builder.get_ensemble()
        for algo in ensemble.get_algo_ensemble():
            print(algo[AlgoEnsembleKeys.ID])

    def tearDown(self) -> None:
        set_determinism(None)
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
