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
from typing import TYPE_CHECKING

import numpy as np
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader, create_test_image_2d, create_test_image_3d, pad_list_data_collate
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandAffined,
    RandAxisFlipd,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    ResizeWithPadOrCropd,
)
from monai.utils import optional_import, set_determinism
from tests.utils import make_nifti_image

if TYPE_CHECKING:

    has_nib = True
else:
    _, has_nib = optional_import("nibabel")

KEYS = ["image", "label"]

TESTS_3D = [
    (t.__class__.__name__ + (" pad_list_data_collate" if collate_fn else " default_collate"), t, collate_fn, 3)
    for collate_fn in [None, pad_list_data_collate]
    for t in [
        RandFlipd(keys=KEYS, spatial_axis=[1, 2]),
        RandAxisFlipd(keys=KEYS),
        RandRotate90d(keys=KEYS, spatial_axes=(1, 2)),
        RandZoomd(keys=KEYS, prob=0.5, min_zoom=0.5, max_zoom=1.1, keep_size=True),
        RandRotated(keys=KEYS, range_x=np.pi),
        RandAffined(keys=KEYS, rotate_range=np.pi),
    ]
]

TESTS_2D = [
    (t.__class__.__name__ + (" pad_list_data_collate" if collate_fn else " default_collate"), t, collate_fn, 2)
    for collate_fn in [None, pad_list_data_collate]
    for t in [
        RandFlipd(keys=KEYS, spatial_axis=[1]),
        RandAxisFlipd(keys=KEYS),
        RandRotate90d(keys=KEYS, spatial_axes=(0, 1)),
        RandZoomd(keys=KEYS, prob=0.5, min_zoom=0.5, max_zoom=1.1, keep_size=True),
        RandRotated(keys=KEYS, range_x=np.pi),
        RandAffined(keys=KEYS, rotate_range=np.pi),
    ]
]


class TestInverseCollation(unittest.TestCase):
    """Test collation for of random transformations with prob == 0 and 1."""

    def setUp(self):
        if not has_nib:
            self.skipTest("nibabel required for test_inverse")

        set_determinism(seed=0)

        im_fname, seg_fname = [make_nifti_image(i) for i in create_test_image_3d(101, 100, 107)]
        load_ims = Compose([LoadImaged(KEYS), AddChanneld(KEYS)])
        self.batch_size = 10
        self.data_3d = [load_ims({"image": im_fname, "label": seg_fname}) for _ in range(self.batch_size)]

        im_fname, seg_fname = [make_nifti_image(i) for i in create_test_image_2d(62, 37, rad_max=10)]
        load_ims = Compose([LoadImaged(KEYS), AddChanneld(KEYS)])
        self.batch_size = 7
        self.data_2d = [load_ims({"image": im_fname, "label": seg_fname}) for _ in range(self.batch_size)]

    def tearDown(self):
        set_determinism(seed=None)

    @parameterized.expand(TESTS_2D + TESTS_3D)
    def test_collation(self, _, transform, collate_fn, ndim):
        if ndim == 3:
            data = self.data_3d
        else:
            data = self.data_2d
        if collate_fn:
            modified_transform = transform
        else:
            modified_transform = Compose([transform, ResizeWithPadOrCropd(KEYS, 100)])

        # num workers = 0 for mac
        num_workers = 2 if sys.platform != "darwin" else 0

        dataset = CacheDataset(data, transform=modified_transform, progress=False)
        loader = DataLoader(dataset, num_workers, batch_size=self.batch_size, collate_fn=collate_fn)

        for _ in loader:
            pass


if __name__ == "__main__":
    unittest.main()
