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

from ignite.metrics import Accuracy
import sys
import logging
import numpy as np
import torch
from ignite.engine import create_supervised_evaluator, _prepare_batch
from torch.utils.data import DataLoader

# assumes the framework is found here, change as necessary
sys.path.append("../..")
from monai.handlers.classification_saver import ClassificationSaver
from monai.handlers.checkpoint_loader import CheckpointLoader
from monai.handlers.stats_handler import StatsHandler
from monai.transforms.composables import LoadNiftid, AddChanneld, Rescaled, Resized
import monai.transforms.compose as transforms
import monai

monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# IXI dataset as a demo, dowloadable from https://brain-development.org/ixi-dataset/
images = [
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
# 2 binary labels for gender classification: man and woman
labels = np.array([
    0, 0, 1, 0, 1, 0, 1, 0, 1, 0
])
val_files = [{'img': img, 'label': label} for img, label in zip(images, labels)]

# Define transforms for image
val_transforms = transforms.Compose([
    LoadNiftid(keys=['img']),
    AddChanneld(keys=['img']),
    Rescaled(keys=['img']),
    Resized(keys=['img'], output_spatial_shape=(96, 96, 96))
])

# Create DenseNet121
net = monai.networks.nets.densenet3d.densenet121(
    in_channels=1,
    out_channels=2,
)
device = torch.device("cuda:0")


def prepare_batch(batch, device=None, non_blocking=False):
    return _prepare_batch((batch['img'], batch['label']), device, non_blocking)


metric_name = 'Accuracy'
# add evaluation metric to the evaluator engine
val_metrics = {metric_name: Accuracy()}
# ignite evaluator expects batch=(img, label) and returns output=(y_pred, y) at every iteration,
# user can add output_transform to return other values
evaluator = create_supervised_evaluator(net, val_metrics, device, True, prepare_batch=prepare_batch)

# Add stats event handler to print validation stats via evaluator
val_stats_handler = StatsHandler(
    name='evaluator',
    output_transform=lambda x: None  # no need to print loss value, so disable per iteration output
)
val_stats_handler.attach(evaluator)

# for the arrary data format, assume the 3rd item of batch data is the meta_data
prediction_saver = ClassificationSaver(output_dir='tempdir', name='evaluator',
                                       batch_transform=lambda batch: {'filename_or_obj': batch['img.filename_or_obj']},
                                       output_transform=lambda output: output[0].argmax(1))
prediction_saver.attach(evaluator)

# the model was trained by "densenet_training_dict" exmple
CheckpointLoader(load_path='./runs/net_checkpoint_40.pth', load_dict={'net': net}).attach(evaluator)

# create a validation data loader
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

state = evaluator.run(val_loader)
