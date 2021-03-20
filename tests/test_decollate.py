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
from enum import Enum
from typing import List, Tuple

import numpy as np
import torch
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader, create_test_image_2d
from monai.data.utils import decollate_batch
from monai.transforms import AddChanneld, Compose, LoadImaged, RandFlipd, SpatialPadd, ToTensord
from monai.transforms.post.dictionary import Decollated
from monai.transforms.spatial.dictionary import RandAffined, RandRotate90d
from monai.utils import optional_import, set_determinism
from monai.utils.enums import InverseKeys
from tests.utils import make_nifti_image

_, has_nib = optional_import("nibabel")

KEYS = ["image"]

TESTS: List[Tuple] = []
TESTS.append((SpatialPadd(KEYS, 150), RandFlipd(KEYS, prob=1.0, spatial_axis=1)))
TESTS.append((RandRotate90d(KEYS, prob=0.0, max_k=1),))
TESTS.append((RandAffined(KEYS, prob=0.0, translate_range=10),))


class TestDeCollate(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)

        im = create_test_image_2d(100, 101)[0]
        self.data = [{"image": make_nifti_image(im) if has_nib else im} for _ in range(6)]

    def tearDown(self) -> None:
        set_determinism(None)

    def check_match(self, in1, in2):
        if isinstance(in1, dict):
            self.assertTrue(isinstance(in2, dict))
            for (k1, v1), (k2, v2) in zip(in1.items(), in2.items()):
                if isinstance(k1, Enum) and isinstance(k2, Enum):
                    k1, k2 = k1.value, k2.value
                self.check_match(k1, k2)
                # Transform ids won't match for windows with multiprocessing, so don't check values
                if k1 == InverseKeys.ID.value and sys.platform in ["darwin", "win32"]:
                    continue
                self.check_match(v1, v2)
        elif isinstance(in1, (list, tuple)):
            for l1, l2 in zip(in1, in2):
                self.check_match(l1, l2)
        elif isinstance(in1, (str, int)):
            self.assertEqual(in1, in2)
        elif isinstance(in1, (torch.Tensor, np.ndarray)):
            np.testing.assert_array_equal(in1, in2)
        else:
            raise RuntimeError(f"Not sure how to compare types. type(in1): {type(in1)}, type(in2): {type(in2)}")

    @parameterized.expand(TESTS)
    def test_decollation(self, *transforms):

        batch_size = 2
        num_workers = 2

        t_compose = Compose([AddChanneld(KEYS), Compose(transforms), ToTensord(KEYS)])
        # If nibabel present, read from disk
        if has_nib:
            t_compose = Compose([LoadImaged("image"), t_compose])

        dataset = CacheDataset(self.data, t_compose, progress=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for b, batch_data in enumerate(loader):
            decollated_1 = decollate_batch(batch_data)
            decollated_2 = Decollated()(batch_data)

            for decollated in [decollated_1, decollated_2]:
                for i, d in enumerate(decollated):
                    self.check_match(dataset[b * batch_size + i], d)


if __name__ == "__main__":
    unittest.main()
