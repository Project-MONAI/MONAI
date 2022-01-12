# Copyright (c) MONAI Consortium
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
from monai.transforms import Compose, DataStatsd, Randomizable, SimulateDelayd
from monai.utils import set_determinism
from monai.utils.enums import CommonKeys

TEST_CASE_1 = [[{CommonKeys.IMAGE: np.asarray([1, 2, 3])}, {CommonKeys.IMAGE: np.asarray([4, 5])}]]

TEST_CASE_2 = [[{CommonKeys.LABEL: torch.as_tensor([[3], [2]])}, {CommonKeys.LABEL: np.asarray([[1], [2]])}]]


class TestDataLoader(unittest.TestCase):
    def test_values(self):
        datalist = [
            {CommonKeys.IMAGE: "spleen_19.nii.gz", CommonKeys.LABEL: "spleen_label_19.nii.gz"},
            {CommonKeys.IMAGE: "spleen_31.nii.gz", CommonKeys.LABEL: "spleen_label_31.nii.gz"},
        ]
        transform = Compose(
            [
                DataStatsd(
                    keys=[CommonKeys.IMAGE, CommonKeys.LABEL], data_shape=False, value_range=False, data_value=True
                ),
                SimulateDelayd(keys=[CommonKeys.IMAGE, CommonKeys.LABEL], delay_time=0.1),
            ]
        )
        dataset = CacheDataset(data=datalist, transform=transform, cache_rate=0.5, cache_num=1)
        n_workers = 0 if sys.platform == "win32" else 2
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=n_workers)
        for d in dataloader:
            self.assertEqual(d[CommonKeys.IMAGE][0], "spleen_19.nii.gz")
            self.assertEqual(d[CommonKeys.IMAGE][1], "spleen_31.nii.gz")
            self.assertEqual(d[CommonKeys.LABEL][0], "spleen_label_19.nii.gz")
            self.assertEqual(d[CommonKeys.LABEL][1], "spleen_label_31.nii.gz")

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_exception(self, datalist):
        dataset = Dataset(data=datalist, transform=None)
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)
        with self.assertRaisesRegex((TypeError, RuntimeError), "Collate error on the key"):
            for _ in dataloader:
                pass


class _RandomDataset(torch.utils.data.Dataset, Randomizable):
    def __getitem__(self, index):
        return self.R.randint(0, 1000, (1,))

    def __len__(self):
        return 8


class TestLoaderRandom(unittest.TestCase):
    """
    Testing data loader working with the randomizable interface
    """

    def setUp(self):
        set_determinism(0)

    def tearDown(self):
        set_determinism(None)

    def test_randomize(self):
        dataset = _RandomDataset()
        dataloader = DataLoader(dataset, batch_size=2, num_workers=3)
        output = []
        for _ in range(2):
            for batch in dataloader:
                output.extend(batch.data.numpy().flatten().tolist())
        self.assertListEqual(output, [594, 170, 524, 778, 370, 906, 292, 589, 762, 763, 156, 886, 42, 405, 221, 166])


if __name__ == "__main__":
    unittest.main()
