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
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, _prepare_batch
from ignite.handlers import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

# assumes the framework is found here, change as necessary
sys.path.append("..")
import monai
import monai.transforms.compose as transforms
from monai.transforms.composables import \
    LoadNiftid, AddChanneld, Rescaled, Resized, RandRotate90d
from monai.handlers.stats_handler import StatsHandler
from ignite.metrics import Accuracy
from monai.handlers.utils import stopping_fn_from_metric

monai.config.print_config()

# demo dataset, user can easily change to own dataset
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
train_files = [{'img': img, 'label': label} for img, label in zip(images[:10], labels[:10])]
val_files = [{'img': img, 'label': label} for img, label in zip(images[-10:], labels[-10:])]

# Define transforms for image
train_transforms = transforms.Compose([
    LoadNiftid(keys=['img']),
    AddChanneld(keys=['img']),
    Rescaled(keys=['img']),
    Resized(keys=['img'], output_shape=(96, 96, 96)),
    RandRotate90d(keys=['img'], prob=0.8, axes=[1, 3])
])
val_transforms = transforms.Compose([
    LoadNiftid(keys=['img']),
    AddChanneld(keys=['img']),
    Rescaled(keys=['img']),
    Resized(keys=['img'], output_shape=(96, 96, 96))
])

# Define dataset, dataloader.
check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
check_data = monai.utils.misc.first(check_loader)
print(check_data['img'].shape, check_data['label'])

lr = 1e-5

# Create DenseNet121, CrossEntropyLoss and Adam optimizer.
net = monai.networks.nets.densenet3d.densenet121(
    in_channels=1,
    out_channels=2,
)

loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr)


# Create trainer
def prepare_batch(batch, device=None, non_blocking=False):
    return _prepare_batch((batch['img'], batch['label']), device, non_blocking)


# Create trainer
device = torch.device("cuda:0")
trainer = create_supervised_trainer(net, opt, loss, device, False, prepare_batch=prepare_batch,
                                    output_transform=lambda x, y, y_pred, loss: [y_pred, loss.item(), y])

# adding checkpoint handler to save models (network params and optimizer stats) during training
checkpoint_handler = ModelCheckpoint('./runs/', 'net', n_saved=10, require_empty=False)
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                          handler=checkpoint_handler,
                          to_save={'net': net, 'opt': opt})
train_stats_handler = StatsHandler()
train_stats_handler.attach(trainer)

# Set parameters for validation
validation_every_n_epochs = 1
metric_name = 'Accuracy'

# add evaluation metric to the evaluator engine
val_metrics = {metric_name: Accuracy()}
evaluator = create_supervised_evaluator(net, val_metrics, device, True, prepare_batch=prepare_batch)

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
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())


@trainer.on(Events.EPOCH_COMPLETED(every=validation_every_n_epochs))
def run_validation(engine):
    evaluator.run(val_loader)

# create a training data loader
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

train_epochs = 30
state = trainer.run(train_loader, train_epochs)
