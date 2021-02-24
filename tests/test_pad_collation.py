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
from parameterized import parameterized

from monai.data.utils import pad_list_data_collate
from monai.transforms import RandRotate90d, RandRotated, RandSpatialCropd, RandZoomd
from monai.utils import set_determinism


from monai.data import CacheDataset, DataLoader

TESTS: List[Tuple] = []

TESTS.append((RandSpatialCropd("image", roi_size=[8, 7], random_size=True),))
TESTS.append((RandRotated("image", prob=1, range_x=np.pi, keep_size=False),))
TESTS.append((RandZoomd("image", prob=1, min_zoom=1.1, max_zoom=2.0, keep_size=False),))
TESTS.append((RandRotate90d("image", prob=1, max_k=2),))


class TestPadCollation(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)
        # image is non square to throw rotation errors
        im = np.arange(0, 10 * 9).reshape(1, 10, 9)
        self.data = [{"image": im} for _ in range(20)]

    def tearDown(self) -> None:
        set_determinism(None)

    @parameterized.expand(TESTS)
    def test_pad_collation(self, transform):

        dataset = CacheDataset(self.data, transform, progress=False)

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
