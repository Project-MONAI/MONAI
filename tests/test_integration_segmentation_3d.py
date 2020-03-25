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
import monai.transforms.compose as transforms
from monai.data.nifti_saver import NiftiSaver
from monai.data.synthetic import create_test_image_3d
from monai.data.utils import list_data_collate
from monai.metrics.compute_meandice import compute_meandice
from monai.networks.nets.unet import UNet
from monai.transforms.composables import (AsChannelFirstd, LoadNiftid, RandCropByPosNegLabeld, RandRotate90d, Rescaled)
from monai.utils.sliding_window_inference import sliding_window_inference
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

from tests.utils import skip_if_quick


def run_training_test(root_dir, device=torch.device("cuda:0")):
    monai.config.print_config()
    images = sorted(glob(os.path.join(root_dir, 'img*.nii.gz')))
    segs = sorted(glob(os.path.join(root_dir, 'seg*.nii.gz')))
    train_files = [{'img': img, 'seg': seg} for img, seg in zip(images[:20], segs[:20])]
    val_files = [{'img': img, 'seg': seg} for img, seg in zip(images[-20:], segs[-20:])]

    # define transforms for image and segmentation
    train_transforms = transforms.Compose([
        LoadNiftid(keys=['img', 'seg']),
        AsChannelFirstd(keys=['img', 'seg'], channel_dim=-1),
        Rescaled(keys=['img', 'seg']),
        RandCropByPosNegLabeld(keys=['img', 'seg'], label_key='seg', size=[96, 96, 96], pos=1, neg=1, num_samples=4),
        RandRotate90d(keys=['img', 'seg'], prob=0.8, spatial_axes=[0, 2])
    ])
    train_transforms.set_random_state(1234)
    val_transforms = transforms.Compose([
        LoadNiftid(keys=['img', 'seg']),
        AsChannelFirstd(keys=['img', 'seg'], channel_dim=-1),
        Rescaled(keys=['img', 'seg'])
    ])
    val_transforms.set_random_state(1234)

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds,
                              batch_size=2,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=list_data_collate,
                            pin_memory=torch.cuda.is_available())

    # create UNet, DiceLoss and Adam optimizer
    model = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = monai.losses.DiceLoss(do_sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)

    # start a typical PyTorch training
    val_interval = 2
    best_metric, best_metric_epoch = -1, -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter(log_dir=os.path.join(root_dir, 'runs'))
    model_filename = os.path.join(root_dir, 'best_metric_model.pth')
    for epoch in range(6):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, 5))
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (batch_data['img'].to(device), batch_data['seg'].to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print("%d/%d, train_loss:%0.4f" % (step, epoch_len, loss.item()))
            writer.add_scalar('train_loss', loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print("epoch %d average loss:%0.4f" % (epoch + 1, epoch_loss))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images = val_data['img']
                    val_labels = val_data['seg']
                    sw_batch_size, roi_size = 4, (96, 96, 96)
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model, device)
                    value = compute_meandice(y_pred=val_outputs,
                                             y=val_labels.to(device),
                                             include_background=True,
                                             to_onehot_y=False,
                                             mutually_exclusive=False)
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_filename)
                    print('saved new best metric model')
                print("current epoch %d current mean dice: %0.4f best mean dice: %0.4f at epoch %d" %
                      (epoch + 1, metric, best_metric, best_metric_epoch))
                writer.add_scalar('val_mean_dice', metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag='image')
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag='label')
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag='output')
    print('train completed, best_metric: %0.4f  at epoch: %d' % (best_metric, best_metric_epoch))
    writer.close()
    return epoch_loss_values, best_metric, best_metric_epoch


def run_inference_test(root_dir, device=torch.device("cuda:0")):
    images = sorted(glob(os.path.join(root_dir, 'im*.nii.gz')))
    segs = sorted(glob(os.path.join(root_dir, 'seg*.nii.gz')))
    val_files = [{'img': img, 'seg': seg} for img, seg in zip(images, segs)]

    # define transforms for image and segmentation
    val_transforms = transforms.Compose([
        LoadNiftid(keys=['img', 'seg']),
        AsChannelFirstd(keys=['img', 'seg'], channel_dim=-1),
        Rescaled(keys=['img', 'seg'])
    ])
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    # sliding window inferene need to input 1 image in every iteration
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=list_data_collate,
                            pin_memory=torch.cuda.is_available())

    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model_filename = os.path.join(root_dir, 'best_metric_model.pth')
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    with torch.no_grad():
        metric_sum = 0.
        metric_count = 0
        saver = NiftiSaver(output_dir=os.path.join(root_dir, 'output'), dtype=int)
        for val_data in val_loader:
            # define sliding window size and batch size for windows inference
            sw_batch_size, roi_size = 4, (96, 96, 96)
            val_outputs = sliding_window_inference(val_data['img'], roi_size, sw_batch_size, model, device)
            val_labels = val_data['seg'].to(device)
            value = compute_meandice(y_pred=val_outputs,
                                     y=val_labels,
                                     include_background=True,
                                     to_onehot_y=False,
                                     mutually_exclusive=False)
            metric_count += len(value)
            metric_sum += value.sum().item()
            saver.save_batch(
                val_outputs, {
                    'filename_or_obj': val_data['img.filename_or_obj'], 'original_affine':
                        val_data['img.original_affine'], 'affine': val_data['img.affine']
                })
        metric = metric_sum / metric_count
    return metric


class IntegrationSegmentation3D(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        self.data_dir = tempfile.mkdtemp()
        for i in range(40):
            im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, 'img%i.nii.gz' % i))
            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(self.data_dir, 'seg%i.nii.gz' % i))

        np.random.seed(seed=None)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    @skip_if_quick
    def test_training(self):
        losses, best_metric, best_metric_epoch = run_training_test(self.data_dir, device=self.device)

        # check training properties
        np.testing.assert_allclose(losses, [
            0.5241468191146851, 0.4485286593437195, 0.42851402163505553, 0.4130884766578674, 0.39990419149398804,
            0.38985557556152345
        ], rtol=1e-5)
        np.testing.assert_allclose(best_metric, 0.9660249322652816, rtol=1e-5)
        np.testing.assert_allclose(best_metric_epoch, 4)
        self.assertTrue(len(glob(os.path.join(self.data_dir, 'runs'))) > 0)
        model_file = os.path.join(self.data_dir, 'best_metric_model.pth')
        self.assertTrue(os.path.exists(model_file))

        infer_metric = run_inference_test(self.data_dir, device=self.device)

        # check inference properties
        np.testing.assert_allclose(infer_metric, 0.9674960002303123, rtol=1e-5)
        output_files = sorted(glob(os.path.join(self.data_dir, 'output', 'img*', '*.nii.gz')))
        sums = [616752.0, 642981.0, 653042.0, 615904.0, 651592.0, 680353.0, 648408.0, 670216.0, 693561.0, 746859.0,
                678080.0, 603877.0, 653672.0, 559537.0, 669992.0, 663388.0, 705862.0, 564044.0, 656242.0, 697152.0,
                726184.0, 698474.0, 701097.0, 600841.0, 681251.0, 652593.0, 717659.0, 701682.0, 597122.0, 542172.0,
                582078.0, 627985.0, 598525.0, 649180.0, 639703.0, 656896.0, 696359.0, 660675.0, 643457.0, 506309.0]
        for (output, s) in zip(output_files, sums):
            np.testing.assert_allclose(np.sum(nib.load(output).get_fdata()), s, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
