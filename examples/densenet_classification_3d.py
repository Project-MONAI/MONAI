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

import sys
import logging
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
from monai.transforms import (AddChannel, Rescale, ToTensor, UniformRandomPatch)
from monai.handlers.stats_handler import StatsHandler
from ignite.metrics import Accuracy
from monai.handlers.utils import stopping_fn_from_metric

monai.config.print_config()

# FIXME: temp test dataset, Wenqi will replace later
images = [
    "/workspace/data/medical/ixi/IXI-T1/IXI314-IOP-0889-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI249-Guys-1072-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI609-HH-2600-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI173-HH-1590-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI020-Guys-0700-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI342-Guys-0909-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI134-Guys-0780-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI577-HH-2661-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI066-Guys-0731-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI130-HH-1528-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI607-Guys-1097-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI175-HH-1570-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI385-HH-2078-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI344-Guys-0905-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI409-Guys-0960-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI584-Guys-1129-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI253-HH-1694-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI092-HH-1436-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI574-IOP-1156-T1.nii.gz",
    "/workspace/data/medical/ixi/IXI-T1/IXI585-Guys-1130-T1.nii.gz"
]
labels = np.array([
    0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0
])

# Define transforms for image and segmentation
imtrans = transforms.Compose([
    Rescale(),
    AddChannel(),
    UniformRandomPatch((96, 96, 96)),
    ToTensor()
])

# Define nifti dataset, dataloader.
ds = NiftiDataset(image_files=images, labels=labels, transform=imtrans)
loader = DataLoader(ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
im, label = monai.utils.misc.first(loader)
print(type(im), im.shape, label)

lr = 1e-5

# Create DenseNet121, CrossEntropyLoss and Adam optimizer.
net = monai.networks.nets.densenet3d.densenet121(
    in_channels=1,
    out_channels=2,
)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr)

# Create trainer
device = torch.device("cuda:0")
trainer = create_supervised_trainer(net, opt, loss, device, False)

# adding checkpoint handler to save models (network params and optimizer stats) during training
checkpoint_handler = ModelCheckpoint('./runs/', 'net', n_saved=10, require_empty=False)
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                          handler=checkpoint_handler,
                          to_save={'net': net, 'opt': opt})
train_stats_handler = StatsHandler()
train_stats_handler.attach(trainer)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(engine):
    engine.logger.info("Epoch[%s] Loss: %s", engine.state.epoch, engine.state.output)

# Set parameters for validation
validation_every_n_epochs = 1
metric_name = 'Accuracy'

# add evaluation metric to the evaluator engine
val_metrics = {metric_name: Accuracy()}
evaluator = create_supervised_evaluator(net, val_metrics, device, True)

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
val_ds = NiftiDataset(image_files=images[-5:], labels=labels[-5:], transform=imtrans)
val_loader = DataLoader(ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())


@trainer.on(Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
def run_validation(engine):
    evaluator.run(val_loader)

# create a training data loader
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

train_ds = NiftiDataset(image_files=images[:15], labels=labels[:15], transform=imtrans)
train_loader = DataLoader(train_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())

train_epochs = 30
state = trainer.run(train_loader, train_epochs)
