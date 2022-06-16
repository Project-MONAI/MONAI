# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.data.image_writer import ITKWriter
from monai.transforms import EnsureChannelFirst, LoadImage, Randomizable, Transform
from monai.utils import convert_data_type, optional_import
from tests.utils import download_url_or_skip_test, testing_data_config

itk, has_itk = optional_import("itk")
TINY_DIFF = 1e-4

key = "ref_avg152T1_LR"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", f"{key}.nii.gz")

TEST_CASES = [
    ["BorderPad", {"spatial_border": (2, 3, 4)}, TINY_DIFF],
    ["CenterScaleCrop", {"roi_scale": (0.6, 0.7, 0.8)}, TINY_DIFF],
    ["CenterSpatialCrop", {"roi_size": (30, 200, 52)}, TINY_DIFF],
    ["CropForeground", {"k_divisible": 3}, TINY_DIFF],
    ["DivisiblePad", {"k": 16}, TINY_DIFF],
    ["RandScaleCrop", {"roi_scale": (0.3, 0.4, 0.5)}, TINY_DIFF],
    ["RandSpatialCrop", {"roi_size": (31, 32, 33)}, TINY_DIFF],
    ["ResizeWithPadOrCrop", {"spatial_size": (50, 80, 200)}, TINY_DIFF],
    ["Spacing", {"pixdim": (1.0, 1.1, 1.2)}, TINY_DIFF],
    ["Orientation", {"axcodes": "RAS"}, TINY_DIFF],
    ["Flip", {"spatial_axis": (0, 1)}, TINY_DIFF],
    ["Resize", {"spatial_size": (100, 201, 134)}, 10.0],
    ["Rotate", {"angle": (np.pi / 3, np.pi / 2, np.pi / 4), "mode": "bilinear"}, 20.0],
    ["Zoom", {"zoom": (0.8, 0.91, 1.2)}, 20.0],
    ["Rotate90", {"k": 3}, TINY_DIFF],
    ["RandRotate90", {"prob": 1.0, "max_k": 3}, TINY_DIFF],
    ["RandRotate", {"prob": 1.0, "range_x": np.pi / 3}, 20.0],
    ["RandFlip", {"prob": 1.0}, TINY_DIFF],
    ["RandAxisFlip", {"prob": 1.0}, TINY_DIFF],
    ["RandZoom", {"prob": 1.0, "mode": "trilinear"}, TINY_DIFF],
    [
        "Affine",
        {
            "rotate_params": (np.pi / 4, 0.0, 0.0),
            "translate_params": (3, 4, 5),
            "spatial_size": (30, 40, 50),
            "mode": "bilinear",
            "image_only": True,
        },
        20.0,
    ],
    [
        "RandAffine",
        {
            "prob": 1.0,
            "rotate_range": (np.pi / 4, np.pi / 3, np.pi / 2),
            "translate_range": (3, 4, 5),
            "scale_range": (-0.1, 0.2),
            "spatial_size": (30, 40, 50),
            "cache_grid": True,
            "mode": "bilinear",
        },
        20.0,
    ],
]


def _create_itk_obj(array, affine):
    itk_img = deepcopy(array)
    itk_img = convert_data_type(itk_img, np.ndarray)[0]
    itk_obj = ITKWriter.create_backend_obj(itk_img, channel_dim=None, affine=affine, affine_lps=True)
    return itk_obj


def _resample_to_affine(itk_obj, ref_obj):
    """linear resample"""
    dim = itk_obj.GetImageDimension()
    transform = itk.IdentityTransform[itk.D, dim].New()
    interpolator = itk.LinearInterpolateImageFunction[type(itk_obj), itk.D].New()
    resampled = itk.resample_image_filter(
        Input=itk_obj, interpolator=interpolator, transform=transform, UseReferenceImage=True, ReferenceImage=ref_obj
    )
    return resampled


@unittest.skipUnless(has_itk, "Requires itk package.")
class TestAffineConsistencyITK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        config = testing_data_config("images", f"{key}")
        download_url_or_skip_test(url=config["url"], filepath=FILE_PATH)

    @parameterized.expand(TEST_CASES)
    def test_linear_consistent(self, xform_cls, input_dict, atol):
        """xformcls testing itk consistency"""
        img = LoadImage()(FILE_PATH)
        img = EnsureChannelFirst()(img)

        ref_1 = _create_itk_obj(img[0], img.affine)

        if isinstance(xform_cls, Transform):
            xform = xform_cls
            output = xform(img, **input_dict)
        else:
            if isinstance(xform_cls, str):
                xform_cls, _ = optional_import("monai.transforms", name=xform_cls)
            xform = xform_cls(**input_dict)
            if isinstance(xform, Randomizable):
                xform.set_random_state(5)
            output = xform(img)

        ref_2 = _create_itk_obj(output[0], output.affine)
        expected = _resample_to_affine(ref_1, ref_2)

        # compare ref_2 and expected results from itk
        diff = np.abs(itk.GetArrayFromImage(ref_2) - itk.GetArrayFromImage(expected))
        avg_diff = np.mean(diff)

        self.assertTrue(avg_diff < atol, f"{xform_cls} avg_diff: {avg_diff}, tol: {atol}")


if __name__ == "__main__":
    unittest.main()
