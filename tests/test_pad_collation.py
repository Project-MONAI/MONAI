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

import random
import unittest
from typing import List, Tuple

import numpy as np
import torch
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
    RandRotate,
    RandRotate90,
    RandRotate90d,
    RandRotated,
    RandSpatialCrop,
    RandSpatialCropd,
    RandZoom,
    RandZoomd,
)
from monai.utils import set_determinism

TESTS: List[Tuple] = []


TESTS.append((dict, RandSpatialCropd("image", roi_size=[8, 7], random_size=True)))
TESTS.append((dict, RandRotated("image", prob=1, range_x=np.pi, keep_size=False)))
TESTS.append((dict, RandZoomd("image", prob=1, min_zoom=1.1, max_zoom=2.0, keep_size=False)))
TESTS.append((dict, RandRotate90d("image", prob=1, max_k=2)))

TESTS.append((list, RandSpatialCrop(roi_size=[8, 7], random_size=True)))
TESTS.append((list, RandRotate(prob=1, range_x=np.pi, keep_size=False)))
TESTS.append((list, RandZoom(prob=1, min_zoom=1.1, max_zoom=2.0, keep_size=False)))
TESTS.append((list, RandRotate90(prob=1, max_k=2)))


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transforms):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transforms(self.images[index]), self.labels[index]


class TestPadCollation(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)
        # image is non square to throw rotation errors
        im = np.arange(0, 10 * 9).reshape(1, 10, 9)
        num_elements = 20
        self.dict_data = [{"image": im} for _ in range(num_elements)]
        self.list_data = [im for _ in range(num_elements)]
        self.list_labels = [random.randint(0, 1) for _ in range(num_elements)]

    def tearDown(self) -> None:
        set_determinism(None)

    @parameterized.expand(TESTS)
    def test_pad_collation(self, t_type, transform):

        if t_type == dict:
            dataset = CacheDataset(self.dict_data, transform, progress=False)
        else:
            dataset = _Dataset(self.list_data, self.list_labels, transform)

        # Default collation should raise an error
        loader_fail = DataLoader(dataset, batch_size=10)
        with self.assertRaises(RuntimeError):
            for _ in loader_fail:
                pass

        # Padded collation shouldn't
        loader = DataLoader(dataset, batch_size=2, collate_fn=pad_list_data_collate)
        for _ in loader:
            pass


if __name__ == "__main__":
    unittest.main()
