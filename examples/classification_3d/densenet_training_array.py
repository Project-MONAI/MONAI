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

import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
    images = [
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI314-IOP-0889-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI249-Guys-1072-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI609-HH-2600-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI173-HH-1590-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI020-Guys-0700-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI342-Guys-0909-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI134-Guys-0780-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI577-HH-2661-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI066-Guys-0731-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI130-HH-1528-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI607-Guys-1097-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI175-HH-1570-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI385-HH-2078-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI344-Guys-0905-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI409-Guys-0960-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI584-Guys-1129-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI253-HH-1694-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI092-HH-1436-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI574-IOP-1156-T1.nii.gz"]),
        os.sep.join(["workspace", "data", "medical", "ixi", "IXI-T1", "IXI585-Guys-1130-T1.nii.gz"]),
    ]

    # 2 binary labels for gender classification: man and woman
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    # Define transforms
    train_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), RandRotate90(), ToTensor()])
    val_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), ToTensor()])

    # Define nifti dataset, data loader
    check_ds = NiftiDataset(image_files=images, labels=labels, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())
    im, label = monai.utils.misc.first(check_loader)
    print(type(im), im.shape, label)

    # create a training data loader
    train_ds = NiftiDataset(image_files=images[:10], labels=labels[:10], transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = NiftiDataset(image_files=images[-10:], labels=labels[-10:], transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                metric = num_correct / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
