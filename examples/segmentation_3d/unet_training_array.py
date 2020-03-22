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
import logging
import nibabel as nib
import numpy as np
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

import monai
import monai.transforms.compose as transforms

from monai.data.nifti_reader import NiftiDataset
from monai.transforms import AddChannel, Rescale, RandUniformPatch, Resize
from monai.handlers.stats_handler import StatsHandler
from monai.handlers.tensorboard_handlers import TensorBoardStatsHandler, TensorBoardImageHandler
from monai.handlers.mean_dice import MeanDice
from monai.data.synthetic import create_test_image_3d
from monai.handlers.utils import stopping_fn_from_metric
from monai.networks.utils import predict_segmentation

monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Create a temporary directory and 40 random image, mask paris
tempdir = tempfile.mkdtemp()
print('generating synthetic data to {} (this may take a while)'.format(tempdir))
for i in range(40):
    im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1)

    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'im%i.nii.gz' % i))

    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'seg%i.nii.gz' % i))

images = sorted(glob(os.path.join(tempdir, 'im*.nii.gz')))
segs = sorted(glob(os.path.join(tempdir, 'seg*.nii.gz')))

# Define transforms for image and segmentation
train_imtrans = transforms.Compose([
    Rescale(),
    AddChannel(),
    RandUniformPatch((96, 96, 96))
])
train_segtrans = transforms.Compose([
    AddChannel(),
    RandUniformPatch((96, 96, 96))
])
val_imtrans = transforms.Compose([
    Rescale(),
    AddChannel(),
    Resize((96, 96, 96))
])
val_segtrans = transforms.Compose([
    AddChannel(),
    Resize((96, 96, 96))
])

# Define nifti dataset, dataloader
check_ds = NiftiDataset(images, segs, transform=train_imtrans, seg_transform=train_segtrans)
check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, seg = monai.utils.misc.first(check_loader)
print(im.shape, seg.shape)

# create a training data loader
train_ds = NiftiDataset(images[:20], segs[:20], transform=train_imtrans, seg_transform=train_segtrans)
train_loader = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
# create a validation data loader
val_ds = NiftiDataset(images[-20:], segs[-20:], transform=val_imtrans, seg_transform=val_segtrans)
val_loader = DataLoader(val_ds, batch_size=5, num_workers=8, pin_memory=torch.cuda.is_available())

# Create UNet, DiceLoss and Adam optimizer
net = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
loss = monai.losses.DiceLoss(do_sigmoid=True)
lr = 1e-3
opt = torch.optim.Adam(net.parameters(), lr)
device = torch.device("cuda:0")

# ignite trainer expects batch=(img, seg) and returns output=loss at every iteration,
# user can add output_transform to return other values, like: y_pred, y, etc.
trainer = create_supervised_trainer(net, opt, loss, device, False)

# adding checkpoint handler to save models (network params and optimizer stats) during training
checkpoint_handler = ModelCheckpoint('./runs/', 'net', n_saved=10, require_empty=False)
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                          handler=checkpoint_handler,
                          to_save={'net': net, 'opt': opt})

# StatsHandler prints loss at every iteration and print metrics at every epoch,
# we don't set metrics for trainer here, so just print loss, user can also customize print functions
# and can use output_transform to convert engine.state.output if it's not a loss value
train_stats_handler = StatsHandler(name='trainer')
train_stats_handler.attach(trainer)

# TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
train_tensorboard_stats_handler = TensorBoardStatsHandler()
train_tensorboard_stats_handler.attach(trainer)

validation_every_n_epochs = 1
# Set parameters for validation
metric_name = 'Mean_Dice'
# add evaluation metric to the evaluator engine
val_metrics = {metric_name: MeanDice(add_sigmoid=True, to_onehot_y=False)}

# ignite evaluator expects batch=(img, seg) and returns output=(y_pred, y) at every iteration,
# user can add output_transform to return other values
evaluator = create_supervised_evaluator(net, val_metrics, device, True)


@trainer.on(Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
def run_validation(engine):
    evaluator.run(val_loader)


# Add early stopping handler to evaluator
early_stopper = EarlyStopping(patience=4,
                              score_function=stopping_fn_from_metric(metric_name),
                              trainer=trainer)
evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=early_stopper)

# Add stats event handler to print validation stats via evaluator
val_stats_handler = StatsHandler(
    name='evaluator',
    output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
    global_epoch_transform=lambda x: trainer.state.epoch)  # fetch global epoch number from trainer
val_stats_handler.attach(evaluator)

# add handler to record metrics to TensorBoard at every validation epoch
val_tensorboard_stats_handler = TensorBoardStatsHandler(
    output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
    global_epoch_transform=lambda x: trainer.state.epoch)  # fetch global epoch number from trainer
val_tensorboard_stats_handler.attach(evaluator)

# add handler to draw the first image and the corresponding label and model output in the last batch
# here we draw the 3D output as GIF format along Depth axis, at every validation epoch
val_tensorboard_image_handler = TensorBoardImageHandler(
    batch_transform=lambda batch: (batch[0], batch[1]),
    output_transform=lambda output: predict_segmentation(output[0]),
    global_iter_transform=lambda x: trainer.state.epoch
)
evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=val_tensorboard_image_handler)

train_epochs = 30
state = trainer.run(train_loader, train_epochs)
