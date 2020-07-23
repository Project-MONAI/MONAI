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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiSaver, create_test_image_3d, list_data_collate
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
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"]),
            ScaleIntensityd(keys=["img", "seg"]),
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
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"]),
            ScaleIntensityd(keys=["img", "seg"]),
            ToTensord(keys=["img", "seg"]),
        ]
    )

    # create a training data loader
    if cachedataset:
        train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.8)
    else:
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=list_data_collate)
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
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
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], mode=["bilinear", "nearest"]),
            ScaleIntensityd(keys=["img", "seg"]),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # sliding window inferene need to input 1 image in every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
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
        saver = NiftiSaver(output_dir=os.path.join(root_dir, "output"), dtype=int)
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
            torch.manual_seed(0)

            repeated.append([])
            losses, best_metric, best_metric_epoch = run_training_test(
                self.data_dir, device=self.device, cachedataset=(i == 2)
            )

            # check training properties
            np.testing.assert_allclose(
                losses,
                [
                    0.5446726709604264,
                    0.4751088947057724,
                    0.4449631154537201,
                    0.42703594267368317,
                    0.4333809733390808,
                    0.42501968443393706,
                ],
                rtol=1e-3,
            )
            repeated[i].extend(losses)
            print("best metric", best_metric)
            np.testing.assert_allclose(best_metric, 0.9315427839756012, rtol=1e-3)
            repeated[i].append(best_metric)
            np.testing.assert_allclose(best_metric_epoch, 6)
            self.assertTrue(len(glob(os.path.join(self.data_dir, "runs"))) > 0)
            model_file = os.path.join(self.data_dir, "best_metric_model.pth")
            self.assertTrue(os.path.exists(model_file))

            infer_metric = run_inference_test(self.data_dir, device=self.device)

            # check inference properties
            print("infer metric", infer_metric)
            np.testing.assert_allclose(infer_metric, 0.9317556858062744, rtol=1e-3)
            repeated[i].append(infer_metric)
            output_files = sorted(glob(os.path.join(self.data_dir, "output", "img*", "*.nii.gz")))
            sums = [
                0.12219095230102539,
                0.13068485260009766,
                0.13124942779541016,
                0.12073802947998047,
                0.16392803192138672,
                0.14664554595947266,
                0.12586545944213867,
                0.14537811279296875,
                0.13469552993774414,
                0.1546192169189453,
                0.13963079452514648,
                0.14705324172973633,
                0.12376976013183594,
                0.0970311164855957,
                0.13933563232421875,
                0.17511940002441406,
                0.1516733169555664,
                0.08219337463378906,
                0.16858196258544922,
                0.17458581924438477,
                0.17008686065673828,
                0.18051767349243164,
                0.14070510864257812,
                0.1136326789855957,
                0.12689924240112305,
                0.12335014343261719,
                0.20095491409301758,
                0.13935327529907227,
                0.12905502319335938,
                0.08764886856079102,
                0.10255098342895508,
                0.11131525039672852,
                0.09727668762207031,
                0.13176202774047852,
                0.14140892028808594,
                0.16825628280639648,
                0.19260501861572266,
                0.15650701522827148,
                0.16521501541137695,
                0.0629119873046875,
            ]
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                np.testing.assert_allclose(ave, s, rtol=5e-3)
                repeated[i].append(ave)
        np.testing.assert_allclose(repeated[0], repeated[1])
        np.testing.assert_allclose(repeated[0], repeated[2])


if __name__ == "__main__":
    unittest.main()
