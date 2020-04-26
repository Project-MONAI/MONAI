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

from monai import config
from monai.transforms import Compose, AddChannel, ScaleIntensity, ToTensor
from monai.networks.nets import UNet
from monai.data import create_test_image_3d, sliding_window_inference, NiftiSaver, NiftiDataset
from monai.metrics import compute_meandice

def main():
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tempdir = tempfile.mkdtemp()
    print('generating synthetic data to {} (this may take a while)'.format(tempdir))
    for i in range(5):
        im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1)

        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, 'im%i.nii.gz' % i))

        n = nib.Nifti1Image(seg, np.eye(4))
        nib.save(n, os.path.join(tempdir, 'seg%i.nii.gz' % i))

    images = sorted(glob(os.path.join(tempdir, 'im*.nii.gz')))
    segs = sorted(glob(os.path.join(tempdir, 'seg*.nii.gz')))

    # define transforms for image and segmentation
    imtrans = Compose([ScaleIntensity(), AddChannel(), ToTensor()])
    segtrans = Compose([AddChannel(), ToTensor()])
    val_ds = NiftiDataset(images, segs, transform=imtrans, seg_transform=segtrans, image_only=False)
    # sliding window inference for one image at every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

    device = torch.device('cuda:0')
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    model.load_state_dict(torch.load('best_metric_model.pth'))
    model.eval()
    with torch.no_grad():
        metric_sum = 0.
        metric_count = 0
        saver = NiftiSaver(output_dir='./output')
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (96, 96, 96)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=True,
                                     to_onehot_y=False, add_sigmoid=True)
            metric_count += len(value)
            metric_sum += value.sum().item()
            val_outputs = (val_outputs.sigmoid() >= 0.5).float()
            saver.save_batch(val_outputs, val_data[2])
        metric = metric_sum / metric_count
        print('evaluation metric:', metric)
    shutil.rmtree(tempdir)

if __name__ == '__main__':
    main()
