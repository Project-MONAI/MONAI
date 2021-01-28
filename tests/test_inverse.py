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

from monai.transforms.transform import InvertibleTransform
import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms import Compose, SpatialPad, SpatialPadd, Rotated, AddChanneld
from monai.data import create_test_image_2d
from monai.utils import Method
import matplotlib.pyplot as plt

TEST_0 = [
    {"image": np.arange(0, 10).reshape(1, 10)},
    [
        SpatialPadd("image", spatial_size=[15]),
        SpatialPadd("image", spatial_size=[21], method=Method.END),
        SpatialPadd("image", spatial_size=[24]),
    ],
]

TEST_1 = [
    {"image": np.arange(0, 10 * 9).reshape(1, 10, 9)},
    [
        SpatialPadd("image", spatial_size=[11, 12]),
        SpatialPadd("image", spatial_size=[12, 21]),
        SpatialPadd("image", spatial_size=[14, 25], method=Method.END),
    ],
]

TEST_2 = [
    {"image": np.arange(0, 10).reshape(1, 10)},
    [
        Compose(
            [
                SpatialPadd("image", spatial_size=[15]),
                SpatialPadd("image", spatial_size=[21]),
                SpatialPadd("image", spatial_size=[24]),
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

TEST_ROTATE = [
    {"image": create_test_image_2d(100, 100)[0]},
    [
        AddChanneld("image"),
        Rotated("image", -np.pi / 6, True, "bilinear", "border", False),
    ]
]

TESTS_LOSSLESS = [TEST_0, TEST_1, TEST_2]
TESTS_LOSSY = [TEST_ROTATE]
TEST_FAILS = [TEST_FAIL_0]


class TestInverse(unittest.TestCase):
    @parameterized.expand(TESTS_LOSSLESS)
    def test_inverse_lossless(self, data, transforms):
        forwards = [data.copy()]

        # Apply forwards
        for t in transforms:
            forwards.append(t(forwards[-1]))

        # Check that error is thrown when inverse are used out of order.
        t = transforms[0] if len(transforms) > 1 else SpatialPadd("image", [10, 5])
        with self.assertRaises(RuntimeError):
            t.inverse(forwards[-1])

        # Apply inverses
        backwards = [forwards[-1].copy()]
        for i, t in enumerate(reversed(transforms)):
            if isinstance(t, InvertibleTransform):
                backwards.append(t.inverse(backwards[-1]))
                self.assertTrue(np.all(backwards[-1]["image"] == forwards[len(forwards) - i - 2]["image"]))

        # Check we got back to beginning
        self.assertTrue(np.all(backwards[-1]["image"] == forwards[0]["image"]))

    def test_inverse_lossy(self, data, transforms):
        forwards = [data.copy()]

        # Apply forwards
        for t in transforms:
            forwards.append(t(forwards[-1]))

        # Check that error is thrown when inverse are used out of order.
        t = SpatialPadd("image", [10, 5])
        with self.assertRaises(RuntimeError):
            t.inverse(forwards[-1])

        # Apply inverses
        backwards = [forwards[-1].copy()]
        for i, t in enumerate(reversed(transforms)):
            if isinstance(t, InvertibleTransform):
                backwards.append(t.inverse(backwards[-1]))
                # self.assertTrue(np.all(backwards[-1]["image"] == forwards[len(forwards) - i - 2]["image"]))

        # Check we got back to beginning
        # self.assertTrue(np.all(backwards[-1]["image"] == forwards[0]["image"]))
        fig, axes = plt.subplots(1, 3)
        pre = forwards[0]["image"]
        post = backwards[-1]["image"][0]
        diff = post - pre
        for i, (im, title) in enumerate(zip([pre, post, diff],["pre", "post", "diff"])):
            ax = axes[i]
            _ = ax.imshow(im)
            ax.set_title(title, fontsize=25)
            ax.axis('off')
        fig.show()
        pass

    @parameterized.expand(TEST_FAILS)
    def test_fail(self, data, transform):
        d = transform(data)
        with self.assertRaises(RuntimeError):
            d = transform.inverse(d)


if __name__ == "__main__":
    # unittest.main()
    a = TestInverse()
    a.test_inverse_lossy(*TEST_ROTATE)
