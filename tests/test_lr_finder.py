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
import unittest

import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset

from monai.optimizers import LRFinder
from monai.networks.nets import densenet121
from tests.utils import skip_if_quick
from monai.utils import set_determinism
from monai.apps import download_and_extract
from urllib.error import ContentTooShortError, HTTPError
from monai.transforms import AddChannel, Compose, LoadImage, RandFlip, RandRotate, RandZoom, ScaleIntensity, ToTensor

TEST_DATA_URL = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
MD5_VALUE = "0bc7306e7427e00ad1c5526a6677552d"
TASK = "integration_classification_2d"

RAND_SEED = 42
random.seed(RAND_SEED)

class MedNISTDataset(Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

device = "cuda" if torch.cuda.is_available() else "cpu"


class TestLRFinder(unittest.TestCase):

    def setUp(self):
        set_determinism(seed=0)

        base_data_dir = os.environ.get("MONAI_DATA_DIRECTORY")
        if not base_data_dir:
            base_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        data_dir = os.path.join(base_data_dir, "MedNIST")
        dataset_file = os.path.join(base_data_dir, "MedNIST.tar.gz")

        if not os.path.exists(data_dir):
            download_and_extract(TEST_DATA_URL, dataset_file, base_data_dir, MD5_VALUE)
            self.assertTrue(os.path.exists(data_dir))

        class_names = sorted((x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))))
        image_files_list_list = [
            [os.path.join(data_dir, class_name, x) for x in sorted(os.listdir(os.path.join(data_dir, class_name)))]
            for class_name in class_names
        ]
        self.image_files, self.image_classes = [], []
        for i, _ in enumerate(class_names):
            self.image_files.extend(image_files_list_list[i])
            self.image_classes.extend([i] * len(image_files_list_list[i]))

        num_to_keep = 20
        c = list(zip(self.image_files, self.image_classes))
        random.shuffle(c)
        self.image_files, self.image_classes = zip(*c[:num_to_keep])
        self.num_classes = len(np.unique(self.image_classes))

        self.train_transforms = Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                ScaleIntensity(),
                RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandFlip(spatial_axis=0, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                ToTensor(),
            ]
        )
        self.train_transforms.set_random_state(RAND_SEED)

    def test_lr_finder(self):

        model = densenet121(spatial_dims=2, in_channels=1, out_channels=self.num_classes).to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        learning_rate = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        train_ds = MedNISTDataset(self.image_files, self.image_classes, self.train_transforms)
        train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)

        lr_finder = LRFinder(model, optimizer, loss_function, device=device)
        lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
        print(lr_finder.get_steepest_gradient()[0])
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset() # to reset the model and optimizer to their initial state


if __name__ == "__main__":
    # unittest.main()
    a = TestLRFinder()
    a.setUp()
    a.test_lr_finder()
