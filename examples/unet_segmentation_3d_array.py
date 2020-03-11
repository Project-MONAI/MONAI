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

# assumes the framework is found here, change as necessary
sys.path.append("..")

import monai
import monai.transforms.compose as transforms

from monai.data.nifti_reader import NiftiDataset
from monai.transforms import AddChannel, Rescale, ToTensor, UniformRandomPatch
from monai.handlers.stats_handler import StatsHandler
from monai.handlers.tensorboard_handlers import TensorBoardStatsHandler, TensorBoardImageHandler
from monai.handlers.mean_dice import MeanDice
from monai.data.synthetic import create_test_image_3d
from monai.handlers.utils import stopping_fn_from_metric

monai.config.print_config()

# Create a temporary directory and 50 random image, mask paris
tempdir = tempfile.mkdtemp()
print('generating synthetic data to {} (this may take a while)'.format(tempdir))
for i in range(50):
    im, seg = create_test_image_3d(128, 128, 128)

    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'im%i.nii.gz' % i))

    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'seg%i.nii.gz' % i))

images = sorted(glob(os.path.join(tempdir, 'im*.nii.gz')))
segs = sorted(glob(os.path.join(tempdir, 'seg*.nii.gz')))

# Define transforms for image and segmentation
imtrans = transforms.Compose([
    Rescale(),
    AddChannel(),
    UniformRandomPatch((96, 96, 96)),
    ToTensor()
])
segtrans = transforms.Compose([
    AddChannel(),
    UniformRandomPatch((96, 96, 96)),
    ToTensor()
])

# Define nifti dataset, dataloader.
ds = NiftiDataset(images, segs, transform=imtrans, seg_transform=segtrans)
loader = DataLoader(ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, seg = monai.utils.misc.first(loader)
print(im.shape, seg.shape)

lr = 1e-5

# Create UNet, DiceLoss and Adam optimizer.
net = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    num_classes=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

loss = monai.losses.DiceLoss(do_sigmoid=True)
opt = torch.optim.Adam(net.parameters(), lr)


# Since network outputs logits and segmentation, we need a custom function.
def _loss_fn(i, j):
    return loss(i[0], j)


# Create trainer
device = torch.device("cpu:0")
trainer = create_supervised_trainer(net, opt, _loss_fn, device, False,
                                    output_transform=lambda x, y, y_pred, loss: [y_pred[1], loss.item(), y])

# adding checkpoint handler to save models (network params and optimizer stats) during training
checkpoint_handler = ModelCheckpoint('./runs/', 'net', n_saved=10, require_empty=False)
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                          handler=checkpoint_handler,
                          to_save={'net': net, 'opt': opt})
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# print training loss to commandline
train_stats_handler = StatsHandler(output_transform=lambda x: x[1])
train_stats_handler.attach(trainer)

# record training loss to TensorBoard at every iteration
train_tensorboard_stats_handler = TensorBoardStatsHandler(
    output_transform=lambda x: {'training_dice_loss': x[1]},  # plot under tag name taining_dice_loss
    global_epoch_transform=lambda x: trainer.state.epoch)
train_tensorboard_stats_handler.attach(trainer)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(engine):
    engine.logger.info("Epoch[%s] Loss: %s", engine.state.epoch, engine.state.output[1])


# Set parameters for validation
validation_every_n_epochs = 1
metric_name = 'Mean_Dice'

# add evaluation metric to the evaluator engine
val_metrics = {metric_name: MeanDice(
    add_sigmoid=True, to_onehot_y=False, output_transform=lambda output: (output[0][0], output[1]))
}
evaluator = create_supervised_evaluator(net, val_metrics, device, True)

# Add stats event handler to print validation stats via evaluator
val_stats_handler = StatsHandler(
    output_transform=lambda x: None,  # disable per iteration output
    global_epoch_transform=lambda x: trainer.state.epoch)
val_stats_handler.attach(evaluator)

# add handler to record metrics to TensorBoard at every epoch
val_tensorboard_stats_handler = TensorBoardStatsHandler(
    output_transform=lambda x: None,  # no iteration plot
    global_epoch_transform=lambda x: trainer.state.epoch)  # use epoch number from trainer
val_tensorboard_stats_handler.attach(evaluator)
# add handler to draw several images and the corresponding labels and model outputs
# here we draw the first 3 images(draw the first channel) as GIF format along Depth axis
val_tensorboard_image_handler = TensorBoardImageHandler(
    batch_transform=lambda batch: (batch[0], batch[1]),
    output_transform=lambda output: output[0][1],
    global_iter_transform=lambda x: trainer.state.epoch
)
evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=val_tensorboard_image_handler)

# Add early stopping handler to evaluator
early_stopper = EarlyStopping(patience=4,
                              score_function=stopping_fn_from_metric(metric_name),
                              trainer=trainer)
evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=early_stopper)

# create a validation data loader
val_ds = NiftiDataset(images[-20:], segs[-20:], transform=imtrans, seg_transform=segtrans)
val_loader = DataLoader(ds, batch_size=5, num_workers=8, pin_memory=torch.cuda.is_available())


@trainer.on(Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
def run_validation(engine):
    evaluator.run(val_loader)


# create a training data loader
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

train_ds = NiftiDataset(images[:20], segs[:20], transform=imtrans, seg_transform=segtrans)
train_loader = DataLoader(train_ds, batch_size=5, num_workers=8, pin_memory=torch.cuda.is_available())

train_epochs = 30
state = trainer.run(train_loader, train_epochs)
