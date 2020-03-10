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
from glob import glob

import nibabel as nib
import numpy as np
import torch
from ignite.engine import Engine
from torch.utils.data import DataLoader

# assumes the framework is found here, change as necessary
sys.path.append("..")

from monai import config
from monai.handlers.checkpoint_loader import CheckpointLoader
from monai.handlers.segmentation_saver import SegmentationSaver
import monai.transforms.compose as transforms
from monai.data.nifti_reader import NiftiDataset
from monai.transforms import AddChannel, Rescale, ToTensor
from monai.networks.nets.unet import UNet
from monai.networks.utils import predict_segmentation
from monai.data.synthetic import create_test_image_3d
from monai.utils.sliding_window_inference import sliding_window_inference

config.print_config()

tempdir = tempfile.mkdtemp()
# tempdir = './temp'
print('generating synthetic data to {} (this may take a while)'.format(tempdir))
for i in range(50):
    im, seg = create_test_image_3d(256, 256, 256)

    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'im%i.nii.gz' % i))

    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'seg%i.nii.gz' % i))

images = sorted(glob(os.path.join(tempdir, 'im*.nii.gz')))
segs = sorted(glob(os.path.join(tempdir, 'seg*.nii.gz')))
imtrans = transforms.Compose([Rescale(), AddChannel(), ToTensor()])
segtrans = transforms.Compose([AddChannel(), ToTensor()])
ds = NiftiDataset(images, segs, transform=imtrans, seg_transform=segtrans, image_only=False)

device = torch.device("cuda:0")
roi_size = (64, 64, 64)
sw_batch_size = 4
net = UNet(
    dimensions=3,
    in_channels=1,
    num_classes=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
net.to(device)


def _sliding_window_processor(engine, batch):
    net.eval()
    img, seg, meta_data = batch
    with torch.no_grad():
        seg_probs = sliding_window_inference(img, roi_size, sw_batch_size, lambda x: net(x)[0], device)
        return predict_segmentation(seg_probs)


infer_engine = Engine(_sliding_window_processor)

# for the arrary data format, assume the 3rd item of batch data is the meta_data
SegmentationSaver(output_path='tempdir', output_ext='.nii.gz', output_postfix='seg',
                  batch_transform=lambda x: x[2]).attach(infer_engine)
# the model was trained by "unet_segmentation_3d_array" exmple
CheckpointLoader(load_path='./runs/net_checkpoint_120.pth', load_dict={'net': net}).attach(infer_engine)

loader = DataLoader(ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
state = infer_engine.run(loader)
