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
import shutil
import tempfile
import unittest
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiSaver, create_test_image_3d
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsChannelFirstd,
    Compose,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
)
from monai.utils import set_determinism
from monai.visualize import plot_2d_or_3d_image
from tests.utils import skip_if_quick


def run_training_test(root_dir, device=torch.device("cuda:0"), cachedataset=False):
    monai.config.print_config()
    images = sorted(glob(os.path.join(root_dir, "img*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:20], segs[:20])]
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-20:], segs[-20:])]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadNiftid(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            # resampling with align_corners=True or dtype=float64 will generate
            # slight different results between PyTorch 1.5 an 1.6
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"], dtype=np.float32),
            ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.8, spatial_axes=[0, 2]),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    train_transforms.set_random_state(1234)
    val_transforms = Compose(
        [
            LoadNiftid(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            # resampling with align_corners=True or dtype=float64 will generate
            # slight different results between PyTorch 1.5 an 1.6
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"], dtype=np.float32),
            ScaleIntensityd(keys="img"),
            ToTensord(keys=["img", "seg"]),
        ]
    )

    # create a training data loader
    if cachedataset:
        train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.8)
    else:
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = monai.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)
    dice_metric = DiceMetric(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")

    # create UNet, DiceLoss and Adam optimizer
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)

    # start a typical PyTorch training
    val_interval = 2
    best_metric, best_metric_epoch = -1, -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(log_dir=os.path.join(root_dir, "runs"))
    model_filename = os.path.join(root_dir, "best_metric_model.pth")
    for epoch in range(6):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{6}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss:{loss.item():0.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch +1} average loss:{epoch_loss:0.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    sw_batch_size, roi_size = 4, (96, 96, 96)
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    value = dice_metric(y_pred=val_outputs, y=val_labels)
                    not_nans = dice_metric.not_nans.item()
                    metric_count += not_nans
                    metric_sum += value.item() * not_nans
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_filename)
                    print("saved new best metric model")
                print(
                    f"current epoch {epoch +1} current mean dice: {metric:0.4f} "
                    f"best mean dice: {best_metric:0.4f} at epoch {best_metric_epoch}"
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
    print(f"train completed, best_metric: {best_metric:0.4f}  at epoch: {best_metric_epoch}")
    writer.close()
    return epoch_loss_values, best_metric, best_metric_epoch


def run_inference_test(root_dir, device=torch.device("cuda:0")):
    images = sorted(glob(os.path.join(root_dir, "im*.nii.gz")))
    segs = sorted(glob(os.path.join(root_dir, "seg*.nii.gz")))
    val_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    val_transforms = Compose(
        [
            LoadNiftid(keys=["img", "seg"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            # resampling with align_corners=True or dtype=float64 will generate
            # slight different results between PyTorch 1.5 an 1.6
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"], dtype=np.float32),
            ScaleIntensityd(keys="img"),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # sliding window inferene need to input 1 image in every iteration
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4)
    dice_metric = DiceMetric(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")

    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model_filename = os.path.join(root_dir, "best_metric_model.pth")
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    with torch.no_grad():
        metric_sum = 0.0
        metric_count = 0
        # resampling with align_corners=True or dtype=float64 will generate
        # slight different results between PyTorch 1.5 an 1.6
        saver = NiftiSaver(output_dir=os.path.join(root_dir, "output"), dtype=np.float32)
        for val_data in val_loader:
            val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
            # define sliding window size and batch size for windows inference
            sw_batch_size, roi_size = 4, (96, 96, 96)
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            value = dice_metric(y_pred=val_outputs, y=val_labels)
            not_nans = dice_metric.not_nans.item()
            metric_count += not_nans
            metric_sum += value.item() * not_nans
            val_outputs = (val_outputs.sigmoid() >= 0.5).float()
            saver.save_batch(val_outputs, val_data["img_meta_dict"])
        metric = metric_sum / metric_count
    return metric


class IntegrationSegmentation3D(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=0)

        self.data_dir = tempfile.mkdtemp()
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"img{i:d}.nii.gz"))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, f"seg{i:d}.nii.gz"))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

    def tearDown(self):
        set_determinism(seed=None)
        shutil.rmtree(self.data_dir)

    @skip_if_quick
    def test_training(self):
        repeated = []
        for i in range(3):
            set_determinism(0)

            repeated.append([])
            losses, best_metric, best_metric_epoch = run_training_test(
                self.data_dir, device=self.device, cachedataset=(i == 2)
            )

            # check training properties
            np.testing.assert_allclose(
                losses,
                [
                    0.5451718807220459,
                    0.4684293806552887,
                    0.458230060338974,
                    0.4460775762796402,
                    0.42319557070732117,
                    0.42558601796627044,
                ],
                rtol=1e-3,
            )
            repeated[i].extend(losses)
            print("best metric", best_metric)
            np.testing.assert_allclose(best_metric, 0.9273447662591934, rtol=1e-2)
            repeated[i].append(best_metric)
            self.assertTrue(len(glob(os.path.join(self.data_dir, "runs"))) > 0)
            model_file = os.path.join(self.data_dir, "best_metric_model.pth")
            self.assertTrue(os.path.exists(model_file))

            infer_metric = run_inference_test(self.data_dir, device=self.device)

            # check inference properties
            print("infer metric", infer_metric)
            np.testing.assert_allclose(infer_metric, 0.9288241773843765, rtol=1e-2)
            repeated[i].append(infer_metric)
            output_files = sorted(glob(os.path.join(self.data_dir, "output", "img*", "*.nii.gz")))
            sums = [
                0.14179113881298405,
                0.15182228027306346,
                0.15091091883208324,
                0.13969700791333736,
                0.1881854951622041,
                0.16894831795256016,
                0.14630071228430272,
                0.1677380522878062,
                0.15677249812024976,
                0.17877369447403701,
                0.16230628310987547,
                0.16792025726421544,
                0.14537765225675714,
                0.11749463499462881,
                0.16156660298735095,
                0.20056036991544426,
                0.17548468185493868,
                0.10907505439670125,
                0.1934211498809681,
                0.20270994906544348,
                0.19577334971835575,
                0.20804474878505128,
                0.16165314464930466,
                0.13171867765630846,
                0.14824347466189244,
                0.14211906406182773,
                0.2312783540544959,
                0.16042356746245373,
                0.14797339479027408,
                0.10411061680840934,
                0.11839301656739093,
                0.13191428015306098,
                0.1138409252764266,
                0.15377293099831713,
                0.16297197461144622,
                0.19354791141527333,
                0.2224228768397248,
                0.1805050725047178,
                0.18999981601378746,
                0.09255153058282481,
            ]
            print([np.mean(nib.load(output).get_fdata()) for output in output_files])
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                np.testing.assert_allclose(ave, s, atol=1e-2)
                repeated[i].append(ave)
        np.testing.assert_allclose(repeated[0], repeated[1])
        np.testing.assert_allclose(repeated[0], repeated[2])


if __name__ == "__main__":
    unittest.main()
