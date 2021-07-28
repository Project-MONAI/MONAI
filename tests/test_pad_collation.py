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

import unittest
from typing import List, Tuple

import numpy as np
import torch
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import decollate_batch, pad_list_data_collate
from monai.transforms import (
    PadListDataCollate,
    RandRotate,
    RandRotate90,
    RandRotate90d,
    RandRotated,
    RandSpatialCrop,
    RandSpatialCropd,
    RandZoom,
    RandZoomd,
)
from tests.utils import TEST_NDARRAYS

TESTS: List[Tuple] = []

for p in TEST_NDARRAYS:
    for include_label in (True, False):
        for pad_collate in [
            lambda x: pad_list_data_collate(batch=x, method="end", mode="constant", constant_values=1),
            PadListDataCollate(method="end", mode="constant", constant_values=1),
        ]:
            TESTS.append(
                (dict, p, include_label, pad_collate, RandSpatialCropd("im", roi_size=[8, 7], random_size=True))
            )
            TESTS.append(
                (dict, p, include_label, pad_collate, RandRotated("im", prob=1, range_x=np.pi, keep_size=False))
            )
            TESTS.append(
                (
                    dict,
                    p,
                    include_label,
                    pad_collate,
                    RandZoomd("im", prob=1, min_zoom=1.1, max_zoom=2.0, keep_size=False),
                )
            )
            TESTS.append((dict, p, include_label, pad_collate, RandRotate90d("im", prob=1, max_k=2)))

            TESTS.append((list, p, include_label, pad_collate, RandSpatialCrop(roi_size=[8, 7], random_size=True)))
            TESTS.append((list, p, include_label, pad_collate, RandRotate(prob=1, range_x=np.pi, keep_size=False)))
            TESTS.append(
                (list, p, include_label, pad_collate, RandZoom(prob=1, min_zoom=1.1, max_zoom=2.0, keep_size=False))
            )
            TESTS.append((list, p, include_label, pad_collate, RandRotate90(prob=1, max_k=2)))


class TupleDataset(Dataset):
    def __getitem__(self, index):
        return self.transform(self.data[index][0]), self.data[index][1]


class TestPadCollation(unittest.TestCase):
    @staticmethod
    def get_data(t_type, im_type, include_label):
        # image is non square to throw rotation errors
        im = im_type(np.arange(0, 10 * 9).reshape(1, 10, 9))
        num_elements = 20
        out = []
        for _ in range(num_elements):
            label = np.random.randint(0, 1)
            if t_type is dict:
                out.append({"im": im, "label": label} if include_label else {"im": im})
            else:
                out.append((im, label) if include_label else im)
        return out

    @parameterized.expand(TESTS)
    def test_pad_collation(self, t_type, im_type, include_label, collate_method, transform):

        input_data = self.get_data(t_type, im_type, include_label)

        if t_type is dict:
            dataset = CacheDataset(input_data, transform, progress=False)
        elif isinstance(input_data[0], tuple):
            dataset = TupleDataset(input_data, transform)
        else:
            dataset = Dataset(input_data, transform)

        # Default collation should raise an error
        loader_fail = DataLoader(dataset, batch_size=10)
        with self.assertRaises(RuntimeError):
            for _ in loader_fail:
                pass

        # Padded collation shouldn't
        loader = DataLoader(dataset, batch_size=10, collate_fn=collate_method)
        # check collation in forward direction
        for data in loader:
            d = data["im"] if isinstance(data, dict) else data
            i = input_data[0]["im"] if isinstance(data, dict) else input_data[0]
            if isinstance(i, torch.Tensor):
                self.assertEqual(d.device, i.device)

            decollated_data = decollate_batch(data)
            # if a dictionary, do the inverse
            if t_type is dict:
                for d in decollated_data:
                    PadListDataCollate.inverse(d)


if __name__ == "__main__":
    unittest.main()
