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
from ignite.engine import Events, create_supervised_trainer
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader

from monai import application, networks, utils
from monai.data.readers import NiftiDataset
from monai.data.transforms import (AddChannel, Rescale, ToTensor, UniformRandomPatch)

# assumes the framework is found here, change as necessary
sys.path.append("..")

application.config.print_config()


def create_test_image_3d(height, width, depth, num_objs=12, rad_max=30, noise_max=0.0, num_seg_classes=5):
    '''Return a noisy 3D image and segmentation.'''
    image = np.zeros((width, height, depth))

    for i in range(num_objs):
        x = np.random.randint(rad_max, width - rad_max)
        y = np.random.randint(rad_max, height - rad_max)
        z = np.random.randint(rad_max, depth - rad_max)
        rad = np.random.randint(5, rad_max)
        spy, spx, spz = np.ogrid[-x:width - x, -y:height - y, -z:depth - z]
        circle = (spx * spx + spy * spy + spz * spz) <= rad * rad

        if num_seg_classes > 1:
            image[circle] = np.ceil(np.random.random() * num_seg_classes)
        else:
            image[circle] = np.random.random() * 0.5 + 0.5

    labels = np.ceil(image).astype(np.int32)

    norm = np.random.uniform(0, num_seg_classes * noise_max, size=image.shape)
    noisyimage = utils.arrayutils.rescale_array(np.maximum(image, norm))

    return noisyimage, labels


tempdir = tempfile.mkdtemp()

for i in range(50):
    im, seg = create_test_image_3d(256, 256, 256)

    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'im%i.nii.gz' % i))

    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'seg%i.nii.gz' % i))

images = sorted(glob(os.path.join(tempdir, 'im*.nii.gz')))
segs = sorted(glob(os.path.join(tempdir, 'seg*.nii.gz')))

imtrans = transforms.Compose([Rescale(), AddChannel(), UniformRandomPatch((64, 64, 64)), ToTensor()])

segtrans = transforms.Compose([AddChannel(), UniformRandomPatch((64, 64, 64)), ToTensor()])

ds = NiftiDataset(images, segs, imtrans, segtrans)

loader = DataLoader(ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, seg = utils.mathutils.first(loader)
print(im.shape, seg.shape)

lr = 1e-3

net = networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    num_classes=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

loss = networks.losses.DiceLoss()
opt = torch.optim.Adam(net.parameters(), lr)

train_epochs = 30


def _loss_fn(i, j):
    return loss(i[0], j)


device = torch.device("cuda:0")

trainer = create_supervised_trainer(net, opt, _loss_fn, device, False)

checkpoint_handler = ModelCheckpoint('./', 'net', n_saved=10, save_interval=3, require_empty=False)
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={'net': net})


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(engine):
    print("Epoch", engine.state.epoch, "Loss:", engine.state.output)


loader = DataLoader(ds, batch_size=20, num_workers=8, pin_memory=torch.cuda.is_available())

state = trainer.run(loader, train_epochs)
