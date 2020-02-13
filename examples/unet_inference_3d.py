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
import torchvision.transforms as transforms
from ignite.engine import Engine
from torch.utils.data import DataLoader

from monai import config
from monai.handlers.checkpoint_loader import CheckpointLoader
from monai.handlers.segmentation_saver import SegmentationSaver
from monai.data.nifti_reader import NiftiDataset
from monai.transforms import AddChannel, Rescale, ToTensor
from monai.networks.nets.unet import UNet
from monai.networks.utils import predict_segmentation
from monai.data.synthetic import create_test_image_3d
from monai.utils.sliding_window_inference import sliding_window_inference

sys.path.append("..")  # assumes the framework is found here, change as necessary
config.print_config()

tempdir = tempfile.mkdtemp()
# tempdir = './temp'
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

device = torch.device("cpu:0")
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


def _sliding_window_processor(_engine, batch):
    net.eval()
    img, seg, meta_data = batch
    with torch.no_grad():
        seg_probs = sliding_window_inference(img, roi_size, sw_batch_size, lambda x: net(x)[0], device)
        return predict_segmentation(seg_probs)


infer_engine = Engine(_sliding_window_processor)

# checkpoint_handler = ModelCheckpoint('./', 'net', n_saved=10, save_interval=3, require_empty=False)
# infer_engine.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={'net': net})

SegmentationSaver(output_path='tempdir', output_ext='.nii.gz', output_postfix='seg').attach(infer_engine)
CheckpointLoader(load_path='./net_checkpoint_9.pth', load_dict={'net': net}).attach(infer_engine)

loader = DataLoader(ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
state = infer_engine.run(loader)
