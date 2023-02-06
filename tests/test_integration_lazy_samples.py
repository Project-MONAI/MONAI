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
import shutil
import tempfile
import unittest
from glob import glob

import nibabel as nib
import numpy as np
import torch

import monai
from monai.data import create_test_image_3d
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    IdentityD,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ResizeWithPadOrCropD,
    SaveImage,
    ScaleIntensityd,
)
from monai.utils import optional_import, set_determinism
from tests.utils import DistTestCase, skip_if_quick

SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")

TASK = "integration_segmentation_3d"


def run_training_test(root_dir, device="cuda:0", cachedataset=0, readers=(None, None), num_workers=4, lazy=True):
    print(f"test case: {locals()}")
    images = sorted(glob(os.path.join(root_dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:20], segs[:20])]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"], reader=readers[0], image_only=True),
            EnsureChannelFirstd(keys=["img", "seg"]),
            # Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"], dtype=np.float32),
            Orientationd(keys=["img", "seg"], axcodes="ARS"),
            RandRotate90d(keys=["img", "seg"], prob=1.0, spatial_axes=(1, 2)),
            ScaleIntensityd(keys="img"),
            IdentityD(keys="seg"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[32, 40, 41], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.8, spatial_axes=[0, 2]),
            ResizeWithPadOrCropD(keys=["img", "seg"], spatial_size=[32, 40, 48]),
        ],
        lazy_evaluation=lazy,
        mode=(1, 0),
        padding_mode="constant",
        lazy_keys=("img", "seg"),
        lazy_dtype=(torch.float32, torch.uint8),
    )
    # train_transforms.set_random_state(1234)

    # create a training data loader
    if cachedataset == 2:
        train_ds = monai.data.CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=0.8, runtime_cache="process"
        )
    elif cachedataset == 3:
        train_ds = monai.data.LMDBDataset(data=train_files, transform=train_transforms, cache_dir=root_dir)
    else:
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

    # create UNet, DiceLoss and Adam optimizer
    model = monai.networks.nets.UNet(
        spatial_dims=3, in_channels=1, out_channels=1, channels=(2, 2, 2, 2), strides=(2, 2, 2), num_res_units=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)
    loss_function = monai.losses.DiceLoss(sigmoid=True)

    saver = SaveImage(
        output_dir=os.path.join(root_dir, "output"),
        dtype=np.float32,
        output_ext=".nii.gz",
        output_postfix="seg",
        mode="bilinear",
        resample=False,
        separate_folder=False,
        print_log=False,
    )

    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    _g = torch.Generator()
    _g.manual_seed(0)
    set_determinism(0)
    train_loader = monai.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=num_workers, generator=_g, persistent_workers=num_workers > 0
    )
    all_coords = set()
    for epoch in range(3):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/5")
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss:{loss.item():0.4f}")

            for item, in_img, in_seg in zip(outputs, inputs, labels):  # this decollates the batch
                item.copy_meta_from(in_img)
                np.testing.assert_array_equal(item.pending_operations, [])
                np.testing.assert_array_equal(in_seg.pending_operations, [])
                np.testing.assert_allclose(len(item.applied_operations) > 1, True)
                for idx, n in enumerate(item.applied_operations):  # noqa
                    if n["class"] == "RandCropByPosNegLabel":
                        break
                ops = item.applied_operations[idx]["extra_info"]["extra_info"]["cropped"]
                img_name = os.path.basename(item.meta["filename_or_obj"])
                coords = f"{img_name} - {ops}"
                print(coords)
                np.testing.assert_allclose(coords in all_coords, False)
                all_coords.add(coords)
                saver(item)
                saver(in_seg)
    return ops


@skip_if_quick
class IntegrationLazyResampling(DistTestCase):
    def setUp(self):
        monai.config.print_config()
        set_determinism(seed=0)

        self.data_dir = tempfile.mkdtemp()
        for i in range(2):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"seg{i:d}.nii.gz"))

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

    def tearDown(self):
        set_determinism(seed=None)
        shutil.rmtree(self.data_dir)

    def train_and_infer(self, idx=0):
        results = []
        _readers = (None, None)
        if idx == 1:
            _readers = ("itkreader", "itkreader")
        elif idx == 2:
            _readers = ("itkreader", "nibabelreader")
        results = run_training_test(
            self.data_dir, device=self.device, cachedataset=idx, readers=_readers, num_workers=0, lazy=False
        )
        results_expected = run_training_test(
            self.data_dir, device=self.device, cachedataset=idx, readers=_readers, num_workers=2, lazy=True
        )
        np.testing.assert_allclose(results, results_expected)
        return results

    def test_training(self):
        # for i in range(4):
        self.train_and_infer(0)


if __name__ == "__main__":
    unittest.main()
