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
import sys
import unittest
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch
from parameterized import parameterized

from monai.data import CacheDataset, DataLoader, create_test_image_2d, create_test_image_3d
from monai.data.utils import decollate_batch
from monai.networks.nets import UNet
from monai.transforms import (
    AddChannel,
    AddChanneld,
    BorderPadd,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    DivisiblePadd,
    Flipd,
    InvertibleTransform,
    LoadImaged,
    Orientationd,
    Rand2DElasticd,
    Rand3DElasticd,
    RandAffined,
    RandAxisFlipd,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandSpatialCropd,
    RandZoomd,
    Resized,
    ResizeWithPadOrCrop,
    ResizeWithPadOrCropd,
    Rotate90d,
    Rotated,
    Spacingd,
    SpatialCropd,
    SpatialPadd,
    Zoomd,
    allow_missing_keys_mode,
)
from monai.utils import first, optional_import, set_determinism
from monai.utils.enums import InverseKeys
from tests.utils import make_nifti_image, make_rand_affine, test_is_quick

if TYPE_CHECKING:
    has_vtk = True
    has_nib = True
else:
    _, has_vtk = optional_import("vtk")
    _, has_nib = optional_import("nibabel")

KEYS = ["image", "label"]

TESTS: List[Tuple] = []

TESTS.append(
    (
        "SpatialPadd (x2) 2d",
        "2D",
        0.0,
        SpatialPadd(KEYS, spatial_size=[111, 113], method="end"),
        SpatialPadd(KEYS, spatial_size=[118, 117]),
    )
)

TESTS.append(
    (
        "SpatialPadd 3d",
        "3D",
        0.0,
        SpatialPadd(KEYS, spatial_size=[112, 113, 116]),
    )
)

TESTS.append(
    (
        "RandRotated, prob 0",
        "2D",
        0,
        RandRotated(KEYS, prob=0),
    )
)

TESTS.append(
    (
        "SpatialCropd 2d",
        "2D",
        3e-2,
        SpatialCropd(KEYS, [49, 51], [90, 89]),
    )
)

TESTS.append(
    (
        "SpatialCropd 3d",
        "3D",
        4e-2,
        SpatialCropd(KEYS, [49, 51, 44], [90, 89, 93]),
    )
)

TESTS.append(("RandSpatialCropd 2d", "2D", 5e-2, RandSpatialCropd(KEYS, [96, 93], True, False)))

TESTS.append(("RandSpatialCropd 3d", "3D", 2e-2, RandSpatialCropd(KEYS, [96, 93, 92], False, False)))

TESTS.append(
    (
        "BorderPadd 2d",
        "2D",
        0,
        BorderPadd(KEYS, [3, 7, 2, 5]),
    )
)

TESTS.append(
    (
        "BorderPadd 2d",
        "2D",
        0,
        BorderPadd(KEYS, [3, 7]),
    )
)

TESTS.append(
    (
        "BorderPadd 3d",
        "3D",
        0,
        BorderPadd(KEYS, [4]),
    )
)

TESTS.append(
    (
        "DivisiblePadd 2d",
        "2D",
        0,
        DivisiblePadd(KEYS, k=4),
    )
)

TESTS.append(
    (
        "DivisiblePadd 3d",
        "3D",
        0,
        DivisiblePadd(KEYS, k=[4, 8, 11]),
    )
)

TESTS.append(
    (
        "Flipd 3d",
        "3D",
        0,
        Flipd(KEYS, [1, 2]),
    )
)

TESTS.append(
    (
        "Flipd 3d",
        "3D",
        0,
        Flipd(KEYS, [1, 2]),
    )
)

TESTS.append(
    (
        "RandFlipd 3d",
        "3D",
        0,
        RandFlipd(KEYS, 1, [1, 2]),
    )
)

TESTS.append(
    (
        "RandAxisFlipd 3d",
        "3D",
        0,
        RandAxisFlipd(KEYS, 1),
    )
)


TESTS.append(
    (
        "Rotated 2d",
        "2D",
        8e-2,
        Rotated(KEYS, random.uniform(np.pi / 6, np.pi), keep_size=True, align_corners=False),
    )
)

TESTS.append(
    (
        "Rotated 3d",
        "3D",
        1e-1,
        Rotated(KEYS, [random.uniform(np.pi / 6, np.pi) for _ in range(3)], True),  # type: ignore
    )
)

TESTS.append(
    (
        "RandRotated 3d",
        "3D",
        1e-1,
        RandRotated(KEYS, *[random.uniform(np.pi / 6, np.pi) for _ in range(3)], 1),  # type: ignore
    )
)

TESTS.append(
    (
        "Orientationd 3d",
        "3D",
        0,
        # For data loader, output needs to be same size, so input must be square/cubic
        SpatialPadd(KEYS, 110),
        Orientationd(KEYS, "RAS"),
    )
)

TESTS.append(
    (
        "Rotate90d 2d",
        "2D",
        0,
        Rotate90d(KEYS),
    )
)

TESTS.append(
    (
        "Rotate90d 3d",
        "3D",
        0,
        Rotate90d(KEYS, k=2, spatial_axes=(1, 2)),
    )
)

TESTS.append(
    (
        "RandRotate90d 3d",
        "3D",
        0,
        # For data loader, output needs to be same size, so input must be square/cubic
        SpatialPadd(KEYS, 110),
        RandRotate90d(KEYS, prob=1, spatial_axes=(1, 2)),
    )
)

TESTS.append(
    (
        "Zoomd 1d",
        "1D",
        0,
        Zoomd(KEYS, zoom=2, keep_size=False),
    )
)

TESTS.append(
    (
        "Zoomd 2d",
        "2D",
        2e-1,
        Zoomd(KEYS, zoom=0.9),
    )
)

TESTS.append(
    (
        "Zoomd 3d",
        "3D",
        3e-2,
        Zoomd(KEYS, zoom=[2.5, 1, 3], keep_size=False),
    )
)

TESTS.append(("RandZoom 3d", "3D", 9e-2, RandZoomd(KEYS, 1, [0.5, 0.6, 0.9], [1.1, 1, 1.05], keep_size=True)))

TESTS.append(
    (
        "CenterSpatialCropd 2d",
        "2D",
        0,
        CenterSpatialCropd(KEYS, roi_size=95),
    )
)

TESTS.append(
    (
        "CenterSpatialCropd 3d",
        "3D",
        0,
        CenterSpatialCropd(KEYS, roi_size=[95, 97, 98]),
    )
)

TESTS.append(("CropForegroundd 2d", "2D", 0, CropForegroundd(KEYS, source_key="label", margin=2)))

TESTS.append(("CropForegroundd 3d", "3D", 0, CropForegroundd(KEYS, source_key="label")))

TESTS.append(("Spacingd 3d", "3D", 3e-2, Spacingd(KEYS, [0.5, 0.7, 0.9], diagonal=False)))

TESTS.append(("Resized 2d", "2D", 2e-1, Resized(KEYS, [50, 47])))

TESTS.append(("Resized 3d", "3D", 5e-2, Resized(KEYS, [201, 150, 78])))

TESTS.append(("ResizeWithPadOrCropd 3d", "3D", 3e-2, ResizeWithPadOrCropd(KEYS, [201, 150, 78])))

TESTS.append(
    (
        "RandAffine 3d",
        "3D",
        1e-1,
        RandAffined(
            KEYS,
            [155, 179, 192],
            prob=1,
            padding_mode="zeros",
            rotate_range=[np.pi / 6, -np.pi / 5, np.pi / 7],
            shear_range=[(0.5, 0.5)],
            translate_range=[10, 5, -4],
            scale_range=[(0.8, 1.2), (0.9, 1.3)],
        ),
    )
)

if has_vtk:
    TESTS.append(
        (
            "Rand2DElasticd 2d",
            "2D",
            2e-1,
            Rand2DElasticd(
                KEYS,
                spacing=(10.0, 10.0),
                magnitude_range=(1, 1),
                spatial_size=(155, 192),
                prob=1,
                padding_mode="zeros",
                rotate_range=[(np.pi / 6, np.pi / 6)],
                shear_range=[(0.5, 0.5)],
                translate_range=[10, 5],
                scale_range=[(0.2, 0.2), (0.3, 0.3)],
            ),
        )
    )

if not test_is_quick and has_vtk:
    TESTS.append(
        (
            "Rand3DElasticd 3d",
            "3D",
            1e-1,
            Rand3DElasticd(
                KEYS,
                sigma_range=(3, 5),
                magnitude_range=(100, 100),
                prob=1,
                padding_mode="zeros",
                rotate_range=[np.pi / 6, np.pi / 7],
                shear_range=[(0.5, 0.5), 0.2],
                translate_range=[10, 5, 3],
                scale_range=[(0.8, 1.2), (0.9, 1.3)],
            ),
        )
    )

TESTS_COMPOSE_X2 = [(t[0] + " Compose", t[1], t[2], Compose(Compose(t[3:]))) for t in TESTS]

TESTS = TESTS + TESTS_COMPOSE_X2  # type: ignore


class TestInverse(unittest.TestCase):
    """Test inverse methods.
    If tests are failing, the following function might be useful for displaying
    `x`, `fx`, `f⁻¹fx` and `x - f⁻¹fx`.
    .. code-block:: python
        def plot_im(orig, fwd_bck, fwd):
            import matplotlib.pyplot as plt
            diff_orig_fwd_bck = orig - fwd_bck
            ims_to_show = [orig, fwd, fwd_bck, diff_orig_fwd_bck]
            titles = ["x", "fx", "f⁻¹fx", "x - f⁻¹fx"]
            fig, axes = plt.subplots(1, 4, gridspec_kw={"width_ratios": [i.shape[1] for i in ims_to_show]})
            vmin = min(np.array(i).min() for i in [orig, fwd_bck, fwd])
            vmax = max(np.array(i).max() for i in [orig, fwd_bck, fwd])
            for im, title, ax in zip(ims_to_show, titles, axes):
                _vmin, _vmax = (vmin, vmax) if id(im) != id(diff_orig_fwd_bck) else (None, None)
                im = np.squeeze(np.array(im))
                while im.ndim > 2:
                    im = im[..., im.shape[-1] // 2]
                im_show = ax.imshow(np.squeeze(im), vmin=_vmin, vmax=_vmax)
                ax.set_title(title, fontsize=25)
                ax.axis("off")
                fig.colorbar(im_show, ax=ax)
            plt.show()
    This can then be added to the exception:
    .. code-block:: python
        except AssertionError:
            print(
                f"Failed: {name}. Mean diff = {mean_diff} (expected <= {acceptable_diff}), unmodified diff: {unmodded_diff}"
            )
            if orig[0].ndim > 1:
                plot_im(orig, fwd_bck, unmodified)
    """

    def setUp(self):
        if not has_nib:
            self.skipTest("nibabel required for test_inverse")

        set_determinism(seed=0)

        self.all_data = {}

        affine = make_rand_affine()
        affine[0] *= 2

        im_1d = AddChannel()(np.arange(0, 10))
        self.all_data["1D"] = {"image": im_1d, "label": im_1d, "other": im_1d}

        im_2d_fname, seg_2d_fname = [make_nifti_image(i) for i in create_test_image_2d(101, 100)]
        im_3d_fname, seg_3d_fname = [make_nifti_image(i, affine) for i in create_test_image_3d(100, 101, 107)]

        load_ims = Compose([LoadImaged(KEYS), AddChanneld(KEYS)])
        self.all_data["2D"] = load_ims({"image": im_2d_fname, "label": seg_2d_fname})
        self.all_data["3D"] = load_ims({"image": im_3d_fname, "label": seg_3d_fname})

    def tearDown(self):
        set_determinism(seed=None)

    def check_inverse(self, name, keys, orig_d, fwd_bck_d, unmodified_d, acceptable_diff):
        for key in keys:
            orig = orig_d[key]
            fwd_bck = fwd_bck_d[key]
            if isinstance(fwd_bck, torch.Tensor):
                fwd_bck = fwd_bck.cpu().numpy()
            unmodified = unmodified_d[key]
            if isinstance(orig, np.ndarray):
                mean_diff = np.mean(np.abs(orig - fwd_bck))
                unmodded_diff = np.mean(np.abs(orig - ResizeWithPadOrCrop(orig.shape[1:])(unmodified)))
                try:
                    self.assertLessEqual(mean_diff, acceptable_diff)
                except AssertionError:
                    print(
                        f"Failed: {name}. Mean diff = {mean_diff} (expected <= {acceptable_diff}), unmodified diff: {unmodded_diff}"
                    )
                    raise

    @parameterized.expand(TESTS)
    def test_inverse(self, _, data_name, acceptable_diff, *transforms):
        name = _

        data = self.all_data[data_name]

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

    def test_inverse_inferred_seg(self):

        test_data = []
        for _ in range(20):
            image, label = create_test_image_2d(100, 101)
            test_data.append({"image": image, "label": label.astype(np.float32)})

        batch_size = 10
        # num workers = 0 for mac
        num_workers = 2 if sys.platform != "darwin" else 0
        transforms = Compose([AddChanneld(KEYS), SpatialPadd(KEYS, (150, 153)), CenterSpatialCropd(KEYS, (110, 99))])
        num_invertible_transforms = sum(1 for i in transforms.transforms if isinstance(i, InvertibleTransform))

        dataset = CacheDataset(test_data, transform=transforms, progress=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = UNet(
            dimensions=2,
            in_channels=1,
            out_channels=1,
            channels=(2, 4),
            strides=(2,),
        ).to(device)

        data = first(loader)
        labels = data["label"].to(device)
        segs = model(labels).detach().cpu()
        label_transform_key = "label" + InverseKeys.KEY_SUFFIX.value
        segs_dict = {"label": segs, label_transform_key: data[label_transform_key]}

        segs_dict_decollated = decollate_batch(segs_dict)

        # inverse of individual segmentation
        seg_dict = first(segs_dict_decollated)
        with allow_missing_keys_mode(transforms):
            inv_seg = transforms.inverse(seg_dict)["label"]
        self.assertEqual(len(data["label_transforms"]), num_invertible_transforms)
        self.assertEqual(len(seg_dict["label_transforms"]), num_invertible_transforms)
        self.assertEqual(inv_seg.shape[1:], test_data[0]["label"].shape)


if __name__ == "__main__":
    unittest.main()
