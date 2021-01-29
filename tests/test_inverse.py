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
from monai.data import CacheDataset
from monai.data import create_test_image_2d  #, create_test_image_3d
from monai.transforms import AddChanneld, Compose, Rotated, RandRotated, SpatialPad, SpatialPadd
from monai.transforms.transform import InvertibleTransform
from monai.utils import Method, optional_import

# from functools import partial
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
    True,
])

TEST_SPATIALS.append([
    "Spatial 2d",
    {"image": np.arange(0, 10 * 9).reshape(1, 10, 9)},
    [
        SpatialPadd("image", spatial_size=[11, 12]),
        SpatialPadd("image", spatial_size=[12, 21]),
        SpatialPadd("image", spatial_size=[14, 25], method=Method.END),
    ],
    True,
])

TEST_SPATIALS.append([
    "Spatial 3d",
    {"image": np.arange(0, 10 * 9 * 8).reshape(1, 10, 9, 8)},
    [
        SpatialPadd("image", spatial_size=[55, 50, 45]),
    ],
    True
])

TEST_COMPOSES = []
TEST_COMPOSES.append([
    "Compose 2d",
    {
        "image": np.arange(0, 10 * 9).reshape(1, 10, 9),
        "label": np.arange(0, 10 * 9).reshape(1, 10, 9),
        "other": np.arange(0, 10 * 9).reshape(1, 10, 9),
    },
    [
        Compose(
            [
                SpatialPadd(["image", "label"], spatial_size=[15, 12]),
                SpatialPadd(["label"], spatial_size=[21, 32]),
                SpatialPadd(["image"], spatial_size=[55, 50]),
            ]
        )
    ],
    True,
])
TEST_COMPOSES.append([
    "Compose 3d",
    {
        "image": np.arange(0, 10 * 9 * 8).reshape(1, 10, 9, 8),
        "label": np.arange(0, 10 * 9 * 8).reshape(1, 10, 9, 8),
        "other": np.arange(0, 10 * 9 * 8).reshape(1, 10, 9, 8),
    },
    [
        Compose(
            [
                SpatialPadd(["image", "label"], spatial_size=[15, 12, 4]),
                SpatialPadd(["label"], spatial_size=[21, 32, 1]),
                SpatialPadd(["image"], spatial_size=[55, 50, 45]),
            ]
        )
    ],
    True,
])

TEST_FAIL_0 = [
    np.arange(0, 10).reshape(1, 10),
    Compose(
        [
            SpatialPad(spatial_size=[15]),
        ]
    ),
    True,
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
                False,
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
            False,
        ]
        TEST_ROTATES.append(TEST_ROTATE)

TESTS = [*TEST_SPATIALS, *TEST_COMPOSES, *TEST_ROTATES]
TESTS_DATALOADER = [*TEST_COMPOSES, *TEST_SPATIALS]
TESTS_FAIL = [TEST_FAIL_0]


def plot_im(orig, fwd_bck, fwd):
    diff_orig_fwd_bck = orig - fwd_bck
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
        im_show = ax.imshow(np.squeeze(im), vmax=vmax)
        ax.set_title(title, fontsize=25)
        ax.axis("off")
        fig.colorbar(im_show, ax=ax)
    plt.show()

class TestInverse(unittest.TestCase):

    def check_inverse(self, keys, orig_d, fwd_bck_d, unmodified_d, lossless):
        for key in keys:
            orig = orig_d[key]
            fwd_bck = fwd_bck_d[key]
            unmodified = unmodified_d[key]
            try:
                if lossless:
                    self.assertTrue(np.all(orig == fwd_bck))
                else:
                    mean_diff = np.mean(np.abs(orig - fwd_bck))
                    print(f"Mean diff = {mean_diff}")
                    self.assertLess(mean_diff, 1.5e-2)
            except AssertionError:
                if has_matplotlib:
                    plot_im(orig, fwd_bck, unmodified)
                    raise

    # @parameterized.expand(TESTS)
    def test_inverse(self, desc, data, transforms, lossless):
        print(f"testing: {desc} (lossless: {lossless})...")
        forwards = [data.copy()]

        # Apply forwards
        for t in transforms:
            forwards.append(t(forwards[-1]))

        # Check that error is thrown when inverse are used out of order.
        t = SpatialPadd("image", [10, 5])
        with self.assertRaises(RuntimeError):
            t.inverse(forwards[-1])

        # Apply inverses
        fwd_bck = forwards[-1].copy()
        for i, t in enumerate(reversed(transforms)):
            if isinstance(t, InvertibleTransform):
                fwd_bck = t.inverse(fwd_bck)
                self.check_inverse(
                    data.keys(), forwards[- i - 2], fwd_bck,
                    forwards[-1], lossless
                )

    # @parameterized.expand(TESTS_FAIL)
    def test_fail(self, data, transform, _):
        d = transform(data)
        with self.assertRaises(RuntimeError):
            d = transform.inverse(d)

    # @parameterized.expand(TEST_COMPOSES)
    def test_w_data_loader(self, desc, data, transforms, lossless):
        print(f"testing: {desc}...")
        transform = transforms[0]
        numel = 2
        test_data = [data for _ in range(numel)]

        dataset = CacheDataset(data=test_data, transform=transform)
        self.assertEqual(len(dataset), 2)
        num_epochs = 2
        for _ in range(num_epochs):
            for data_fwd in dataset:
                data_fwd_bck = transform.inverse(data_fwd)
                self.check_inverse(
                    data.keys(), data, data_fwd_bck,
                    data_fwd, lossless
                )


if __name__ == "__main__":
    # unittest.main()
    test = TestInverse()
    for t in TESTS:
        test.test_inverse(*t)
    for t in TESTS_DATALOADER:
        test.test_w_data_loader(*t)
    for t in TESTS_FAIL:
        test.test_fail(*t)
