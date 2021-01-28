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

import numpy as np
from parameterized import parameterized

from monai.transforms import Compose, SpatialPad, SpatialPadd

TEST_0 = [
    {"image": np.arange(0, 10).reshape(1, 10)},
    [
        SpatialPadd(keys="image", spatial_size=[15]),
        SpatialPadd(keys="image", spatial_size=[21]),
        SpatialPadd(keys="image", spatial_size=[24]),
    ],
]

TEST_1 = [
    {"image": np.arange(0, 10 * 9).reshape(1, 10, 9)},
    [
        SpatialPadd(keys="image", spatial_size=[11, 12]),
        SpatialPadd(keys="image", spatial_size=[12, 21]),
        SpatialPadd(keys="image", spatial_size=[14, 25]),
    ],
]

TEST_2 = [
    {"image": np.arange(0, 10).reshape(1, 10)},
    [
        Compose(
            [
                SpatialPadd(keys="image", spatial_size=[15]),
                SpatialPadd(keys="image", spatial_size=[21]),
                SpatialPadd(keys="image", spatial_size=[24]),
            ]
        )
    ],
]

TEST_FAIL_0 = [
    np.arange(0, 10).reshape(1, 10),
    Compose(
        [
            SpatialPad(spatial_size=[15]),
        ]
    ),
]

TESTS = [TEST_0, TEST_1, TEST_2]
TEST_FAILS = [TEST_FAIL_0]


class TestInverse(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_inverse(self, data, transforms):
        d = data.copy()

        # Apply forwards
        for t in transforms:
            d = t(d)

        # Check that error is thrown when inverse are used out of order.
        t = transforms[0] if len(transforms) > 1 else SpatialPadd("image", [10, 5])
        with self.assertRaises(RuntimeError):
            t.inverse(d)

        # Apply inverses
        for t in reversed(transforms):
            d = t.inverse(d)

        self.assertTrue(np.all(d["image"] == data["image"]))

    @parameterized.expand(TEST_FAILS)
    def test_fail(self, data, transform):
        d = transform(data)
        with self.assertRaises(RuntimeError):
            d = transform.inverse(d)


if __name__ == "__main__":
    unittest.main()
