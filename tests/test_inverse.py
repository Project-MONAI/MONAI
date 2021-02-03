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
from typing import List, Tuple
from monai.data import create_test_image_2d, create_test_image_3d
from monai.data import CacheDataset
from monai.transforms import (
    InvertibleTransform,
    AddChanneld,
    AddChannel,
    Compose,
    RandRotated,
    RandSpatialCropd,
    Rotated,
    SpatialPad,
    SpatialPadd,
    SpatialCropd,
    BorderPadd,
    DivisiblePadd,
    Flipd,
    LoadImaged,
    Rotate90d,
    Zoomd,
    CenterSpatialCropd,
    CropForegroundd,
    Orientationd,
    Spacingd,
    Resized,
    ResizeWithPadOrCropd,
    RandZoomd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
)
from monai.utils import optional_import, set_determinism
from tests.utils import make_nifti_image, make_rand_affine

# from parameterized import parameterized


if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    has_matplotlib = True
else:
    plt, has_matplotlib = optional_import("matplotlib.pyplot")

# set_determinism(seed=0)

AFFINE = make_rand_affine()
AFFINE[0] *= 2

IM_1D = AddChannel()(np.arange(0, 10))
IM_2D_FNAME, SEG_2D_FNAME = [make_nifti_image(i) for i in create_test_image_2d(100, 101)]
IM_3D_FNAME, SEG_3D_FNAME = [make_nifti_image(i, AFFINE) for i in create_test_image_3d(100, 101, 107)]

KEYS = ["image", "label"]
DATA_1D = {"image": IM_1D, "label": IM_1D, "other": IM_1D}
LOAD_IMS = Compose([LoadImaged(KEYS), AddChanneld(KEYS)])
DATA_2D = LOAD_IMS({"image": IM_2D_FNAME, "label": SEG_2D_FNAME})
DATA_3D = LOAD_IMS({"image": IM_3D_FNAME, "label": SEG_3D_FNAME})

TESTS: List[Tuple] = []

TESTS.append((
    "SpatialPadd (x2) 2d",
    DATA_2D,
    0.0,
    SpatialPadd(KEYS, spatial_size=[111, 113], method="end"),
    SpatialPadd(KEYS, spatial_size=[118, 117]),
))

TESTS.append((
    "SpatialPadd 3d",
    DATA_3D,
    0.0,
    SpatialPadd(KEYS, spatial_size=[112, 113, 116]),
))

TESTS.append((
    "RandRotated, prob 0",
    DATA_2D,
    0,
    RandRotated(KEYS, prob=0),
))

TESTS.append((
    "SpatialCropd 2d",
    DATA_2D,
    3e-2,
    SpatialCropd(KEYS, [49, 51], [90, 89]),
))

TESTS.append((
    "SpatialCropd 3d",
    DATA_3D,
    4e-2,
    SpatialCropd(KEYS, [49, 51, 44], [90, 89, 93]),
))

TESTS.append((
    "RandSpatialCropd 2d",
    DATA_2D,
    5e-2,
    RandSpatialCropd(KEYS, [96, 93], True, False)
))

TESTS.append((
    "RandSpatialCropd 3d",
    DATA_3D,
    2e-2,
    RandSpatialCropd(KEYS, [96, 93, 92], False, True)
))

TESTS.append((
    "BorderPadd 2d",
    DATA_2D,
    0,
    BorderPadd(KEYS, [3, 7, 2, 5]),
))

TESTS.append((
    "BorderPadd 2d",
    DATA_2D,
    0,
    BorderPadd(KEYS, [3, 7]),
))

TESTS.append((
    "BorderPadd 3d",
    DATA_3D,
    0,
    BorderPadd(KEYS, [4]),
))

TESTS.append((
    "DivisiblePadd 2d",
    DATA_2D,
    0,
    DivisiblePadd(KEYS, k=4),
))

TESTS.append((
    "DivisiblePadd 3d",
    DATA_3D,
    0,
    DivisiblePadd(KEYS, k=[4, 8, 11]),
))

TESTS.append((
    "Flipd 3d",
    DATA_3D,
    0,
    Flipd(KEYS, [1, 2]),
))

TESTS.append((
    "Flipd 3d",
    DATA_3D,
    0,
    Flipd(KEYS, [1, 2]),
))

TESTS.append((
    "RandFlipd 3d",
    DATA_3D,
    0,
    RandFlipd(KEYS, 1, [1, 2]),
))

TESTS.append((
    "Rotated 2d",
    DATA_2D,
    8e-2,
    Rotated(KEYS, random.uniform(np.pi / 6, np.pi), keep_size=True, align_corners=False),
))

TESTS.append((
    "Rotated 3d",
    DATA_3D,
    5e-2,
    Rotated(KEYS, [random.uniform(np.pi / 6, np.pi) for _ in range(3)], 1),
))

TESTS.append((
    "RandRotated 3d",
    DATA_3D,
    5e-2,
    RandRotated(KEYS, *(random.uniform(np.pi / 6, np.pi) for _ in range(3)), 1),
))

TESTS.append((
    "Orientationd 3d",
    DATA_3D,
    0,
    Orientationd(KEYS, 'RAS'),
))

TESTS.append((
    "Rotate90d 2d",
    DATA_2D,
    0,
    Rotate90d(KEYS),
))

TESTS.append((
    "Rotate90d 3d",
    DATA_3D,
    0,
    Rotate90d(KEYS, k=2, spatial_axes=[1, 2]),
))

TESTS.append((
    "RandRotate90d 3d",
    DATA_3D,
    0,
    RandRotate90d(KEYS, prob=1, spatial_axes=[1, 2]),
))

TESTS.append((
    "Zoomd 1d",
    DATA_1D,
    0,
    Zoomd(KEYS, zoom=2, keep_size=False),
))

TESTS.append((
    "Zoomd 2d",
    DATA_2D,
    2e-1,
    Zoomd(KEYS, zoom=0.9),
))

TESTS.append((
    "Zoomd 3d",
    DATA_3D,
    3e-2,
    Zoomd(KEYS, zoom=[2.5, 1, 3], keep_size=False),
))

TESTS.append((
    "RandZoom 3d",
    DATA_3D,
    5e-2,
    RandZoomd(KEYS, 1, [0.5, 0.6, 0.9], [3, 4.2, 6.1], keep_size=False)
))

TESTS.append((
    "CenterSpatialCropd 2d",
    DATA_2D,
    0,
    CenterSpatialCropd(KEYS, roi_size=95),
))

TESTS.append((
    "CenterSpatialCropd 3d",
    DATA_3D,
    0,
    CenterSpatialCropd(KEYS, roi_size=[95, 97, 98]),
))

TESTS.append((
    "CropForegroundd 2d",
    DATA_2D,
    0,
    CropForegroundd(KEYS, source_key="label", margin=[2, 1])
))

TESTS.append((
    "CropForegroundd 3d",
    DATA_3D,
    0,
    CropForegroundd(KEYS, source_key="label")
))

TESTS.append((
    "Spacingd 3d",
    DATA_3D,
    3e-2,
    Spacingd(KEYS, [0.5, 0.7, 0.9], diagonal=False)
))

TESTS.append((
    "Resized 2d",
    DATA_2D,
    2e-1,
    Resized(KEYS, [50, 47])
))

TESTS.append((
    "Resized 3d",
    DATA_3D,
    5e-2,
    Resized(KEYS, [201, 150, 78])
))

TESTS.append((
    "ResizeWithPadOrCropd 3d",
    DATA_3D,
    1e-2,
    ResizeWithPadOrCropd(KEYS, [201, 150, 78])
))

TESTS.append((
    "RandAffine 3d",
    DATA_3D,
    5e-2,
    RandAffined(KEYS, [98, 96, 105], 1, rotate_range=np.pi / 6, shear_range=[1, 1, 1], translate_range=[10, 5, -4], scale_range=[0.9, 1, 1.1])
))

TESTS_COMPOSE_X2 = [(t[0] + " Compose", t[1], t[2], Compose(Compose(t[3:]))) for t in TESTS]

TESTS = [*TESTS, *TESTS_COMPOSE_X2]


# Should fail because uses an array transform (SpatialPad), as opposed to dictionary
TEST_FAIL_0 = (DATA_2D["image"], 0.0, Compose([SpatialPad(spatial_size=[101, 103])]))
TESTS_FAIL = [TEST_FAIL_0]

def plot_im(orig, fwd_bck, fwd):
    diff_orig_fwd_bck = orig - fwd_bck
    fig, axes = plt.subplots(
        1, 4, gridspec_kw={"width_ratios": [orig.shape[1], fwd_bck.shape[1], diff_orig_fwd_bck.shape[1], fwd.shape[1]]}
    )
    vmin = min(np.array(i).min() for i in [orig, fwd_bck, fwd])
    vmax = max(np.array(i).max() for i in [orig, fwd_bck, fwd])
    for i, (im, title) in enumerate(
        zip([orig, fwd_bck, diff_orig_fwd_bck, fwd], ["x", "f⁻¹fx", "diff", "fx"])
    ):
        ax = axes[i]
        im = np.squeeze(np.array(im))
        while im.ndim > 2:
            im = im[..., im.shape[-1] // 2]
        _vmin, _vmax = (vmin, vmax) if i != 2 else (None, None)
        im_show = ax.imshow(np.squeeze(im), vmin=_vmin, vmax=_vmax)
        ax.set_title(title, fontsize=25)
        ax.axis("off")
        fig.colorbar(im_show, ax=ax)
    plt.show()


class TestInverse(unittest.TestCase):
    def check_inverse(self, name, keys, orig_d, fwd_bck_d, unmodified_d, acceptable_diff):
        for key in keys:
            orig = orig_d[key]
            fwd_bck = fwd_bck_d[key]
            unmodified = unmodified_d[key]
            if isinstance(orig, np.ndarray):
                mean_diff = np.mean(np.abs(orig - fwd_bck))
                try:
                    self.assertLessEqual(mean_diff, acceptable_diff)
                except AssertionError:
                    print(f"Failed: {name}. Mean diff = {mean_diff} (expected <= {acceptable_diff})")
                    if has_matplotlib:
                        plot_im(orig, fwd_bck, unmodified)
                    raise

    # @parameterized.expand(TESTS)
    def test_inverse(self, _, data, acceptable_diff, *transforms):
        name = _

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
                self.check_inverse(name, data.keys(), forwards[-i - 2], fwd_bck, forwards[-1], acceptable_diff)

    # @parameterized.expand(TESTS_FAIL)
    def test_fail(self, data, _, *transform):
        d = transform[0](data)
        with self.assertRaises(RuntimeError):
            d = transform[0].inverse(d)

    # @parameterized.expand(TEST_COMPOSES)
    def test_w_data_loader(self, _, data, acceptable_diff, *transforms):
        name = _
        transform = transforms[0]
        numel = 2
        test_data = [data for _ in range(numel)]

        dataset = CacheDataset(test_data, transform, progress=False)
        self.assertEqual(len(dataset), 2)
        num_epochs = 2
        for _ in range(num_epochs):
            for data_fwd in dataset:
                data_fwd_bck = transform.inverse(data_fwd)
                self.check_inverse(name, data.keys(), data, data_fwd_bck, data_fwd, acceptable_diff)


if __name__ == "__main__":
    # unittest.main()
    test = TestInverse()
    for t in TESTS:
        test.test_inverse(*t)
        test.test_w_data_loader(*t)
    for t in TESTS_FAIL:
        test.test_fail(*t)
