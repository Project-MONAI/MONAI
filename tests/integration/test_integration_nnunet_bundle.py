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
from pathlib import Path

import numpy as np

from monai.apps.nnunet import nnUNetV2Runner
from monai.apps.nnunet.nnunet_bundle import (
    convert_nnunet_to_monai_bundle,
    get_nnunet_monai_predictor,
    get_nnunet_trainer,
)
from monai.bundle.config_parser import ConfigParser
from monai.data import DataLoader, Dataset, create_test_image_3d
from monai.transforms import Compose, Decollated, EnsureChannelFirstd, LoadImaged, SaveImaged
from monai.utils import optional_import
from tests.test_utils import SkipIfBeforePyTorchVersion, skip_if_downloading_fails, skip_if_no_cuda, skip_if_quick

_, has_tb = optional_import("torch.utils.tensorboard", name="SummaryWriter")
_, has_nnunet = optional_import("nnunetv2")

sim_datalist: dict[str, list[dict]] = {
    "testing": [{"image": "val_001.fake.nii.gz"}, {"image": "val_002.fake.nii.gz"}],
    "training": [
        {"fold": 0, "image": "tr_image_001.fake.nii.gz", "label": "tr_label_001.fake.nii.gz"},
        {"fold": 0, "image": "tr_image_002.fake.nii.gz", "label": "tr_label_002.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_003.fake.nii.gz", "label": "tr_label_003.fake.nii.gz"},
        {"fold": 1, "image": "tr_image_004.fake.nii.gz", "label": "tr_label_004.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_005.fake.nii.gz", "label": "tr_label_005.fake.nii.gz"},
        {"fold": 2, "image": "tr_image_006.fake.nii.gz", "label": "tr_label_006.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_007.fake.nii.gz", "label": "tr_label_007.fake.nii.gz"},
        {"fold": 3, "image": "tr_image_008.fake.nii.gz", "label": "tr_label_008.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_009.fake.nii.gz", "label": "tr_label_009.fake.nii.gz"},
        {"fold": 4, "image": "tr_image_010.fake.nii.gz", "label": "tr_label_010.fake.nii.gz"},
    ],
}


@skip_if_quick
@SkipIfBeforePyTorchVersion((1, 13, 0))
@unittest.skipIf(not has_tb, "no tensorboard summary writer")
@unittest.skipIf(not has_nnunet, "no nnunetv2")
class TestnnUNetBundle(unittest.TestCase):

    def setUp(self) -> None:

        import nibabel as nib

        self.test_dir = tempfile.TemporaryDirectory()
        test_path = self.test_dir.name

        sim_dataroot = os.path.join(test_path, "dataroot")
        if not os.path.isdir(sim_dataroot):
            os.makedirs(sim_dataroot)

        self.sim_dataroot = sim_dataroot
        # Generate a fake dataset
        for d in sim_datalist["testing"] + sim_datalist["training"]:
            im, seg = create_test_image_3d(24, 24, 24, rad_max=10, num_seg_classes=2)
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
        data_src = {"modality": "CT", "datalist": sim_json_datalist, "dataroot": sim_dataroot}

        ConfigParser.export_config_file(data_src, data_src_cfg)
        self.data_src_cfg = data_src_cfg
        self.test_path = test_path

    @skip_if_no_cuda
    def test_nnunet_bundle(self) -> None:
        runner = nnUNetV2Runner(
            input_config=self.data_src_cfg, trainer_class_name="nnUNetTrainer_1epoch", work_dir=self.test_path
        )
        with skip_if_downloading_fails():
            runner.run(run_train=False, run_find_best_configuration=False, run_predict_ensemble_postprocessing=False)

            nnunet_trainer = get_nnunet_trainer(
                dataset_name_or_id=runner.dataset_name, fold=0, configuration="3d_fullres"
            )

            print("Max Epochs: ", nnunet_trainer.num_epochs)
            print("Num Iterations: ", nnunet_trainer.num_iterations_per_epoch)
            print("Train Batch dims: ", next(nnunet_trainer.dataloader_train.generator)["data"].shape)
            print("Val Batch dims: ", next(nnunet_trainer.dataloader_val.generator)["data"].shape)
            print("Network: ", nnunet_trainer.network)
            print("Optimizer: ", nnunet_trainer.optimizer)
            print("Loss Function: ", nnunet_trainer.loss)
            print("LR Scheduler: ", nnunet_trainer.lr_scheduler)
            print("Device: ", nnunet_trainer.device)
            runner.train_single_model("3d_fullres", fold=0)

        nnunet_config = {"dataset_name_or_id": "001", "nnunet_trainer": "nnUNetTrainer_1epoch"}
        self.bundle_root = os.path.join("bundle_root")

        Path(self.bundle_root).joinpath("models").mkdir(parents=True, exist_ok=True)
        convert_nnunet_to_monai_bundle(nnunet_config, self.bundle_root, 0)

        data_transforms = Compose([LoadImaged(keys="image"), EnsureChannelFirstd(keys="image")])
        dataset = Dataset(
            data=[{"image": os.path.join(self.test_path, "dataroot", "val_001.fake.nii.gz")}], transform=data_transforms
        )
        data_loader = DataLoader(dataset, batch_size=1)
        input = next(iter(data_loader))

        predictor = get_nnunet_monai_predictor(Path(self.bundle_root).joinpath("models", "fold_0"))
        pred_batch = predictor(input["image"])
        Path(self.sim_dataroot).joinpath("predictions").mkdir(parents=True, exist_ok=True)

        post_processing_transforms = Compose(
            [
                Decollated(keys=None, detach=True),
                # Not needed after reading the data directly from the MONAI LoadImaged Transform
                # Transposed(keys="pred", indices=[0, 3, 2, 1]),
                SaveImaged(
                    keys="pred", output_dir=Path(self.sim_dataroot).joinpath("predictions"), output_postfix="pred"
                ),
            ]
        )
        post_processing_transforms({"pred": pred_batch})

    def tearDown(self) -> None:
        self.test_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
