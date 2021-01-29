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
from typing import TYPE_CHECKING

import numpy as np

from monai.data import create_test_image_2d
from monai.transforms import AddChanneld, Compose, Rotated, RandRotated, SpatialPad, SpatialPadd
from monai.transforms.transform import InvertibleTransform
from monai.utils import Method, optional_import

# from parameterized import parameterized


if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    has_matplotlib = True
else:
    plt, has_matplotlib = optional_import("matplotlib.pyplot")

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

TEST_ROTATES = []
# for k in [True, False]:
#     for a in [False, True]:
#         TEST_ROTATE = [
#             {"image": create_test_image_2d(100, 100)[0]},
#             [
#                 AddChanneld("image"),
#                 Rotated("image", random.uniform(np.pi / 6, np.pi), k, "bilinear", "border", a),
#             ],
#         ]
#         TEST_ROTATES.append(TEST_ROTATE)
for p in [0, 1]:
    TEST_ROTATE = [
        {"image": create_test_image_2d(100, 100)[0]},
        [
            AddChanneld("image"),
            RandRotated(
                "image",
                random.uniform(np.pi / 6, np.pi),
                random.uniform(np.pi / 6, np.pi),
                random.uniform(np.pi / 6, np.pi),
                p, True, "bilinear", "border", False),
        ],
    ]
    TEST_ROTATES.append(TEST_ROTATE)

TESTS_LOSSLESS = [TEST_0, TEST_1, TEST_2]
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
        im_show = ax.imshow(np.squeeze(im), vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=25)
        ax.axis("off")
        fig.colorbar(im_show, ax=ax)
    plt.show()


class TestInverse(unittest.TestCase):
    # @parameterized.expand(TESTS_LOSSLESS)
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

    # @parameterized.expand(TESTS_LOSSY)
    def test_inverse_lossy(self, data, transforms, visualise=False):
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
                self.assertLess(mean_percent_diff, 10)

        if has_matplotlib and visualise:
            plot_im(forwards[1]["image"], backwards[-1]["image"], forwards[-1]["image"])

        # Check that if the inverse hadn't been called, mean_percent_diff would have been greater
        if forwards[1]["image"].shape == forwards[-1]["image"].shape:
            mean_percent_diff = get_mean_percent_diff(forwards[1]["image"], forwards[-1]["image"])
            self.assertGreater(mean_percent_diff, 50)

    # @parameterized.expand(TESTS_FAIL)
    def test_fail(self, data, transform):
        d = transform(data)
        with self.assertRaises(RuntimeError):
            d = transform.inverse(d)


if __name__ == "__main__":
    # unittest.main()
    test = TestInverse()
    # for t in TESTS_LOSSLESS:
    #     test.test_inverse_lossless(*t)
    for t in TESTS_LOSSY:
        test.test_inverse_lossy(*t, True)
    # for t in TESTS_FAIL:
    #     test.test_fail(*t)
