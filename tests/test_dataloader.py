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

import sys
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import Compose, DataStatsd, SimulateDelayd

TEST_CASE_1 = [
    [
        {"image": np.asarray([1, 2, 3])},
        {"image": np.asarray([4, 5])},
    ]
]

TEST_CASE_2 = [
    [
        {"label": torch.as_tensor([[3], [2]])},
        {"label": np.asarray([[1], [2]])},
    ]
]


class TestDataLoader(unittest.TestCase):
    def test_values(self):
        datalist = [
            {"image": "spleen_19.nii.gz", "label": "spleen_label_19.nii.gz"},
            {"image": "spleen_31.nii.gz", "label": "spleen_label_31.nii.gz"},
        ]
        transform = Compose(
            [
                DataStatsd(keys=["image", "label"], data_shape=False, value_range=False, data_value=True),
                SimulateDelayd(keys=["image", "label"], delay_time=0.1),
            ]
        )
        dataset = CacheDataset(data=datalist, transform=transform, cache_rate=0.5, cache_num=1)
        n_workers = 0 if sys.platform == "win32" else 2
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=n_workers)
        for d in dataloader:
            self.assertEqual(d["image"][0], "spleen_19.nii.gz")
            self.assertEqual(d["image"][1], "spleen_31.nii.gz")
            self.assertEqual(d["label"][0], "spleen_label_19.nii.gz")
            self.assertEqual(d["label"][1], "spleen_label_31.nii.gz")

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_exception(self, datalist):
        dataset = Dataset(data=datalist, transform=None)
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)
        with self.assertRaisesRegex((TypeError, RuntimeError), "Collate error on the key"):
            for _ in dataloader:
                pass


if __name__ == "__main__":
    unittest.main()
