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
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

# assumes the framework is found here, change as necessary
sys.path.append("..")

import monai
import monai.transforms.compose as transforms

from monai.data.nifti_reader import NiftiDataset
from monai.transforms import (AddChannel, Rescale, ToTensor, UniformRandomPatch)
from monai.handlers.stats_handler import StatsHandler
from monai.handlers.mean_dice import MeanDice
from monai.visualize import img2tensorboard
from monai.data.synthetic import create_test_image_3d
from monai.handlers.utils import stopping_fn_from_metric

monai.config.print_config()

# Create a temporary directory and 50 random image, mask paris
tempdir = tempfile.mkdtemp()

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
device = torch.device("cuda:0")
trainer = create_supervised_trainer(net, opt, _loss_fn, device, False,
                                    output_transform=lambda x, y, y_pred, loss: [y_pred, loss.item(), y])

# adding checkpoint handler to save models (network params and optimizer stats) during training
checkpoint_handler = ModelCheckpoint('./runs/', 'net', n_saved=10, require_empty=False)
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                          handler=checkpoint_handler,
                          to_save={'net': net, 'opt': opt})
train_stats_handler = StatsHandler()
train_stats_handler.attach(trainer)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(engine):
    # log loss to tensorboard with second item of engine.state.output, loss.item() from output_transform
    writer.add_scalar('Loss/train', engine.state.output[1], engine.state.epoch)

    # tensor of ones to use where for converting labels to zero and ones
    ones = torch.ones(engine.state.batch[1][0].shape, dtype=torch.int32)
    first_output_tensor = engine.state.output[0][1][0].detach().cpu()
    # log model output to tensorboard, as three dimensional tensor with no channels dimension
    img2tensorboard.add_animated_gif_no_channels(writer, "first_output_final_batch", first_output_tensor, 64,
                                                 255, engine.state.epoch)
    # get label tensor and convert to single class
    first_label_tensor = torch.where(engine.state.batch[1][0] > 0, ones, engine.state.batch[1][0])
    # log label tensor to tensorboard, there is a channel dimension when getting label from batch
    img2tensorboard.add_animated_gif(writer, "first_label_final_batch", first_label_tensor, 64,
                                     255, engine.state.epoch)
    second_output_tensor = engine.state.output[0][1][1].detach().cpu()
    img2tensorboard.add_animated_gif_no_channels(writer, "second_output_final_batch", second_output_tensor, 64,
                                                 255, engine.state.epoch)
    second_label_tensor = torch.where(engine.state.batch[1][1] > 0, ones, engine.state.batch[1][1])
    img2tensorboard.add_animated_gif(writer, "second_label_final_batch", second_label_tensor, 64,
                                     255, engine.state.epoch)
    third_output_tensor = engine.state.output[0][1][2].detach().cpu()
    img2tensorboard.add_animated_gif_no_channels(writer, "third_output_final_batch", third_output_tensor, 64,
                                                 255, engine.state.epoch)
    third_label_tensor = torch.where(engine.state.batch[1][2] > 0, ones, engine.state.batch[1][2])
    img2tensorboard.add_animated_gif(writer, "third_label_final_batch", third_label_tensor, 64,
                                     255, engine.state.epoch)
    engine.logger.info("Epoch[%s] Loss: %s", engine.state.epoch, engine.state.output[1])

writer = SummaryWriter()

# Set parameters for validation
validation_every_n_epochs = 1
metric_name = 'Mean_Dice'

# add evaluation metric to the evaluator engine
val_metrics = {metric_name: MeanDice(add_sigmoid=True)}
evaluator = create_supervised_evaluator(net, val_metrics, device, True,
                                        output_transform=lambda x, y, y_pred: (y_pred[0], y))

# Add stats event handler to print validation stats via evaluator
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
val_stats_handler = StatsHandler()
val_stats_handler.attach(evaluator)

# Add early stopping handler to evaluator.
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

@evaluator.on(Events.EPOCH_COMPLETED)
def log_metrics_to_tensorboard(engine):
    for name, value in engine.state.metrics.items():
        writer.add_scalar('Metrics/{name}', value, trainer.state.epoch)

# create a training data loader
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

train_ds = NiftiDataset(images[:20], segs[:20], transform=imtrans, seg_transform=segtrans)
train_loader = DataLoader(train_ds, batch_size=5, num_workers=8, pin_memory=torch.cuda.is_available())

train_epochs = 30
state = trainer.run(train_loader, train_epochs)
