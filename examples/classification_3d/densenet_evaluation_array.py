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

import monai
from monai.data import CSVSaver, NiftiDataset
from monai.transforms import AddChannel, Compose, Resize, ScaleIntensity, ToTensor


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
    images = [
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
    labels = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    # Define transforms for image
    val_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), ToTensor()])

    # Define nifti dataset
    val_ds = NiftiDataset(image_files=images, labels=labels, transform=val_transforms, image_only=False)
    # create a validation data loader
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

    model.load_state_dict(torch.load("best_metric_model_classification3d_array.pth"))
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        saver = CSVSaver(output_dir="./output")
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_images).argmax(dim=1)
            value = torch.eq(val_outputs, val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
            saver.save_batch(val_outputs, val_data[2])
        metric = num_correct / metric_count
        print("evaluation metric:", metric)
        saver.finalize()


if __name__ == "__main__":
    main()
