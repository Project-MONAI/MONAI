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
import numpy as np
import torch
from torch.utils.data import DataLoader

# assumes the framework is found here, change as necessary
sys.path.append("../..")
import monai
import monai.transforms.compose as transforms
from monai.transforms.composables import \
    LoadNiftid, AddChanneld, ScaleIntensityRanged, RandCropByPosNegLabeld, RandAffined
from monai.data.utils import list_data_collate
from monai.utils.sliding_window_inference import sliding_window_inference
from monai.metrics.compute_meandice import compute_meandice
from datalist import datalist

monai.config.print_config()

# define transforms for image and segmentation
train_transforms = transforms.Compose([
    LoadNiftid(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', size=(96, 96, 96), pos=1, neg=1, num_samples=4),
    RandAffined(keys=['image', 'label'], spatial_size=(96, 96, 96), prob=0.5,
                rotate_range=(np.pi / 10, np.pi / 10, np.pi / 10), scale_range=(0.1, 0.1, 0.1))
])
val_transforms = transforms.Compose([
    LoadNiftid(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True)
])

check_transforms = transforms.Compose([
    LoadNiftid(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True)
])
check_ds = monai.data.Dataset(data=datalist['validation'], transform=check_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = monai.utils.misc.first(check_loader)
print('image shape:', check_data['image'].shape, 'label shape:', check_data['label'].shape)

# create a training data loader
train_ds = monai.data.Dataset(data=datalist['training'], transform=train_transforms)
# use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=2, num_workers=4, collate_fn=list_data_collate)
# create a validation data loader
val_ds = monai.data.Dataset(data=datalist['validation'], transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                 strides=(2, 2, 2, 2), num_res_units=2, instance_norm=False).to(device)
loss_function = monai.losses.DiceLoss(do_softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

best_metric = -1
best_metric_epoch = -1
for epoch in range(100):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch + 1, 100))
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (batch_data['image'].to(device), batch_data['label'].to(device))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print("%d/%d,train_loss:%0.4f" % (step, len(train_ds) // train_loader.batch_size, loss.item()))
    print("epoch %d average loss:%0.4f" % (epoch + 1, epoch_loss / step))
    # do validation every 2 epochs
    if (epoch + 1) % 2 == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.
            metric_count = 0
            for val_data in val_loader:
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_data['image'], roi_size, sw_batch_size, model, device)
                val_labels = val_data['label'].to(device)
                value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False,
                                         to_onehot_y=True, mutually_exclusive=True)
                for batch in value:
                    metric_count += 1
                    metric_sum += batch.item()
            metric = metric_sum / metric_count
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), 'best_metric_model.pth')
                print('saved new best metric model')
            print("current epoch %d current mean dice: %0.4f best mean dice: %0.4f at epoch %d"
                  % (epoch + 1, metric, best_metric, best_metric_epoch))
print('train completed, best_metric: %0.4f  at epoch: %d' % (best_metric, best_metric_epoch))
