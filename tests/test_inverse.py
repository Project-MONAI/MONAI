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

from functools import partial
import random
import unittest
from typing import TYPE_CHECKING

import numpy as np

from monai.data import create_test_image_2d, create_test_image_3d
from monai.transforms import AddChanneld, Compose, Rotated, RandRotated, SpatialPad, SpatialPadd
from monai.transforms.transform import InvertibleTransform
from monai.utils import Method, optional_import

# from parameterized import parameterized


if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    has_matplotlib = True
else:
    plt, has_matplotlib = optional_import("matplotlib.pyplot")

TEST_SPATIALS = []
TEST_SPATIALS.append([
    "Spatial 1d",
    {"image": np.arange(0, 10).reshape(1, 10)},
    [
        SpatialPadd("image", spatial_size=[15]),
        SpatialPadd("image", spatial_size=[21], method=Method.END),
        SpatialPadd("image", spatial_size=[24]),
    ],
])

TEST_SPATIALS.append([
    "Spatial 2d",
    {"image": np.arange(0, 10 * 9).reshape(1, 10, 9)},
    [
        SpatialPadd("image", spatial_size=[11, 12]),
        SpatialPadd("image", spatial_size=[12, 21]),
        SpatialPadd("image", spatial_size=[14, 25], method=Method.END),
    ],
])

TEST_SPATIALS.append([
    "Spatial 3d",
    {"image": np.arange(0, 10 * 9 * 8).reshape(1, 10, 9, 8)},
    [
        SpatialPadd("image", spatial_size=[55, 50, 45]),
    ],
])

TEST_COMPOSE = [
    "Compose",
    {"image": np.arange(0, 10 * 9 * 8).reshape(1, 10, 9, 8)},
    [
        Compose(
            [
                SpatialPadd("image", spatial_size=[15, 12, 4]),
                SpatialPadd("image", spatial_size=[21, 32, 1]),
                SpatialPadd("image", spatial_size=[55, 50, 45]),
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

# TODO: add 3D
TEST_ROTATES = []
for create_im in [create_test_image_2d]:  #, partial(create_test_image_3d, 100)]:
    for keep_size in [True, False]:
        for align_corners in [False, True]:
            im, _ = create_im(100, 100)
            angle = random.uniform(np.pi / 6, np.pi)
            TEST_ROTATE = [
                f"Rotate{im.ndim}d, keep_size={keep_size}, align_corners={align_corners}",
                {"image": im},
                [
                    AddChanneld("image"),
                    Rotated("image", angle, keep_size, "bilinear", "border", align_corners),
                ],
            ]
            TEST_ROTATES.append(TEST_ROTATE)
    for prob in [0, 1]:
        im, _ = create_im(100, 100)
        angles = [random.uniform(np.pi / 6, np.pi) for _ in range(3)]
        TEST_ROTATE = [
            f"RandRotate{im.ndim}d, prob={prob}",
            {"image": im},
            [
                AddChanneld("image"),
                RandRotated("image", *angles, prob, True, "bilinear", "border", False),
            ],
        ]
        TEST_ROTATES.append(TEST_ROTATE)

TESTS_LOSSLESS = [*TEST_SPATIALS, TEST_COMPOSE]
TESTS_LOSSY = [*TEST_ROTATES]
TESTS_FAIL = [TEST_FAIL_0]


def get_percent_diff_im(array_true, array):
    return 100 * (array_true - array) / (array_true + 1e-5)


def get_mean_percent_diff(array_true, array):
    return abs(np.mean(get_percent_diff_im(array_true, array)))


def plot_im(orig, fwd_bck, fwd):
    diff_orig_fwd_bck = get_percent_diff_im(orig, fwd_bck)
    fig, axes = plt.subplots(
        1, 4, gridspec_kw={"width_ratios": [orig.shape[1], fwd_bck.shape[1], diff_orig_fwd_bck.shape[1], fwd.shape[1]]}
    )
    for i, (im, title) in enumerate(
        zip([orig, fwd_bck, diff_orig_fwd_bck, fwd], ["orig", "fwd_bck", "%% diff", "fwd"])
    ):
        ax = axes[i]
        vmax = max(np.max(i) for i in [orig, fwd_bck, fwd]) if i != 2 else None
        im = np.squeeze(im)
        while im.ndim > 2:
            im = im[..., im.shape[-1] // 2]
        im_show = ax.imshow(np.squeeze(im), vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=25)
        ax.axis("off")
        fig.colorbar(im_show, ax=ax)
    plt.show()


class TestInverse(unittest.TestCase):
    # @parameterized.expand(TESTS_LOSSLESS)
    def test_inverse_lossless(self, desc, data, transforms):
        print(f"testing: {desc}...")
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

    # @parameterized.expand(TESTS_LOSSY)
    def test_inverse_lossy(self, desc, data, transforms):
        print("testing: " + desc)
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
                mean_percent_diff = get_mean_percent_diff(backwards[-1]["image"], forwards[-i - 2]["image"])
                try:
                    self.assertLess(mean_percent_diff, 10)
                except AssertionError:
                    if has_matplotlib:
                        plot_im(forwards[1]["image"], backwards[-1]["image"], forwards[-1]["image"])
                        raise

    # @parameterized.expand(TESTS_FAIL)
    def test_fail(self, data, transform):
        d = transform(data)
        with self.assertRaises(RuntimeError):
            d = transform.inverse(d)


if __name__ == "__main__":
    # unittest.main()
    test = TestInverse()
    for t in TESTS_LOSSLESS:
        test.test_inverse_lossless(*t)
    for t in TESTS_LOSSY:
        test.test_inverse_lossy(*t)
    for t in TESTS_FAIL:
        test.test_fail(*t)
