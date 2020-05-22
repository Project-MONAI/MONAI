# Copyright 2020 MONAI Consortium
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
import sys
import tempfile
import shutil
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.transforms import (
    Compose,
    LoadNiftid,
    AsChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ToTensord,
)
from monai.handlers import StatsHandler, ValidationHandler, MeanDice
from monai.data import create_test_image_3d, list_data_collate
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.engines.utils import CommonKeys as Keys


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # create a temporary directory and 40 random image, mask paris
    tempdir = tempfile.mkdtemp()
    print(f"generating synthetic data to {tempdir} (this may take a while)")
    for i in range(40):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"img{i:d}.nii.gz"))
        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
    train_files = [{Keys.IMAGE: img, Keys.LABEL: seg} for img, seg in zip(images[:20], segs[:20])]
    val_files = [{Keys.IMAGE: img, Keys.LABEL: seg} for img, seg in zip(images[-20:], segs[-20:])]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadNiftid(keys=[Keys.IMAGE, Keys.LABEL]),
            AsChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL], channel_dim=-1),
            ScaleIntensityd(keys=[Keys.IMAGE, Keys.LABEL]),
            RandCropByPosNegLabeld(
                keys=[Keys.IMAGE, Keys.LABEL], label_key=Keys.LABEL, size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=[Keys.IMAGE, Keys.LABEL], prob=0.5, spatial_axes=[0, 2]),
            ToTensord(keys=[Keys.IMAGE, Keys.LABEL]),
        ]
    )
    val_transforms = Compose(
        [
            LoadNiftid(keys=[Keys.IMAGE, Keys.LABEL]),
            AsChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL], channel_dim=-1),
            ScaleIntensityd(keys=[Keys.IMAGE, Keys.LABEL]),
            ToTensord(keys=[Keys.IMAGE, Keys.LABEL]),
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=list_data_collate)
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    net = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss = monai.losses.DiceLoss(do_sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)

    val_handlers = [StatsHandler(output_transform=lambda x: None)]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        val_handlers=val_handlers,
        key_val_metric={
            "val_mean_dice": MeanDice(
                include_background=True, add_sigmoid=True, output_transform=lambda x: (x[Keys.PRED], x[Keys.LABEL])
            )
        },
        additional_metrics=None,
    )

    train_handlers = [
        ValidationHandler(validator=evaluator, interval=2, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x[Keys.INFO][Keys.LOSS]),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        train_handlers=train_handlers,
        amp=False,
        key_train_metric=None,
    )
    trainer.run()

    shutil.rmtree(tempdir)


if __name__ == "__main__":
    main()
