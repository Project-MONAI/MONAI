# Copyright 2020 - 2021 MONAI Consortium
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
import time
import unittest

import torch
from torch.utils.data import DataLoader

from monai.apps import MedNISTDataset
from monai.networks.nets import densenet121
from monai.transforms import Compose, CopyToDeviced, ToTensord, LoadImaged, AddChanneld
from tests.utils import skip_if_no_cuda

# This test is only run with cuda
DEVICE = "cuda:0"

@skip_if_no_cuda
class TestDictCopyToDeviceTimeComparison(unittest.TestCase):

    @staticmethod
    def get_data(use_copy_to_device_transform):

        root_dir = os.environ.get("MONAI_DATA_DIRECTORY")
        if not root_dir:
            root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")

        transforms = Compose(
            [
                LoadImaged(keys="image"),
                AddChanneld(keys="image"),
                ToTensord(keys="image"),
            ]
        )
        # If necessary, append the transform
        if use_copy_to_device_transform:
            transforms.transforms = transforms.transforms + (CopyToDeviced(keys="image", device=DEVICE),)

        train_ds = MedNISTDataset(
            root_dir=root_dir,
            transform=transforms,
            section="validation",
            val_frac=0.001,
            download=True,
            num_workers=10,
        )
        train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)
        num_classes = train_ds.get_num_classes()

        model = densenet121(spatial_dims=2, in_channels=1, out_channels=num_classes).to(DEVICE)

        return train_loader, model

    def test_dict_copy_to_device_time_comparison(self):


        for use_copy_transform in [True, False]:
            start_time = time.time()

            train_loader, model = self.get_data(use_copy_transform)

            model.train()
            for batch_data in train_loader:
                inputs, labels = batch_data["image"], batch_data["label"]
                # If using the copy transform, check they're on the GPU
                if use_copy_transform:
                    self.assertEqual(str(inputs.device), DEVICE)
                # Assert not already on device, and then copy them there
                else:
                    self.assertNotEqual(str(inputs.device), DEVICE)
                    inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                loss_function = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), 1e-5)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f"--- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    unittest.main()
