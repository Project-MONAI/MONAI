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
from monai.data import create_test_image_3d, NiftiSaver, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    AsChannelFirstd,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
)
from monai.visualize import plot_2d_or_3d_image
from monai.utils import set_determinism
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
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], interp_order=["bilinear", "nearest"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys=["img", "seg"]),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", size=[96, 96, 96], pos=1, neg=1, num_samples=4
            ),
            RandRotate90d(keys=["img", "seg"], prob=0.8, spatial_axes=[0, 2]),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    train_transforms.set_random_state(1234)
    val_transforms = Compose(
        [
            LoadNiftid(keys=["img", "seg"]),
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], interp_order=["bilinear", "nearest"]),
            AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
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
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available()
    )

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
                    value = compute_meandice(
                        y_pred=val_outputs, y=val_labels, include_background=True, to_onehot_y=False, sigmoid=True
                    )
                    metric_count += len(value)
                    metric_sum += value.sum().item()
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
            Spacingd(keys=["img", "seg"], pixdim=[1.2, 0.8, 0.7], interp_order=["bilinear", "nearest"]),
            ScaleIntensityd(keys=["img", "seg"]),
            ToTensord(keys=["img", "seg"]),
        ]
    )
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # sliding window inferene need to input 1 image in every iteration
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate, pin_memory=torch.cuda.is_available()
    )

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
            value = compute_meandice(
                y_pred=val_outputs, y=val_labels, include_background=True, to_onehot_y=False, sigmoid=True
            )
            metric_count += len(value)
            metric_sum += value.sum().item()
            val_outputs = (val_outputs.sigmoid() >= 0.5).float()
            saver.save_batch(val_outputs, val_data["img_meta"])
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
                    0.5480509608983993,
                    0.46773012578487394,
                    0.4449450016021729,
                    0.4429117202758789,
                    0.4266220510005951,
                    0.4189875662326813,
                ],
                rtol=1e-3,
            )
            repeated[i].extend(losses)
            print("best metric", best_metric)
            np.testing.assert_allclose(best_metric, 0.9281478852033616, rtol=1e-3)
            repeated[i].append(best_metric)
            np.testing.assert_allclose(best_metric_epoch, 6)
            self.assertTrue(len(glob(os.path.join(self.data_dir, "runs"))) > 0)
            model_file = os.path.join(self.data_dir, "best_metric_model.pth")
            self.assertTrue(os.path.exists(model_file))

            infer_metric = run_inference_test(self.data_dir, device=self.device)

            # check inference properties
            print("infer metric", infer_metric)
            np.testing.assert_allclose(infer_metric, 0.9276287078857421, rtol=1e-3)
            repeated[i].append(infer_metric)
            output_files = sorted(glob(os.path.join(self.data_dir, "output", "img*", "*.nii.gz")))
            sums = [
                0.12279891967773438,
                0.13062572479248047,
                0.1316366195678711,
                0.12099170684814453,
                0.16374444961547852,
                0.1474461555480957,
                0.12650299072265625,
                0.14547109603881836,
                0.13530206680297852,
                0.15510225296020508,
                0.1400594711303711,
                0.1470775604248047,
                0.12418031692504883,
                0.09789371490478516,
                0.13944005966186523,
                0.17546558380126953,
                0.1519641876220703,
                0.08295297622680664,
                0.1682891845703125,
                0.17422246932983398,
                0.17064905166625977,
                0.1806168556213379,
                0.14097881317138672,
                0.11398553848266602,
                0.12688827514648438,
                0.1236882209777832,
                0.2005314826965332,
                0.1399993896484375,
                0.12934064865112305,
                0.08836174011230469,
                0.10311698913574219,
                0.11167573928833008,
                0.0979313850402832,
                0.13147354125976562,
                0.14134931564331055,
                0.16820192337036133,
                0.19240856170654297,
                0.1563401222229004,
                0.1651287078857422,
                0.06423425674438477,
            ]
            for (output, s) in zip(output_files, sums):
                ave = np.mean(nib.load(output).get_fdata())
                np.testing.assert_allclose(ave, s, rtol=5e-3)
                repeated[i].append(ave)
        np.testing.assert_allclose(repeated[0], repeated[1])
        np.testing.assert_allclose(repeated[0], repeated[2])


if __name__ == "__main__":
    unittest.main()
