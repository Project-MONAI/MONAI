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

import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms.intensity.dictionary import GetForegroundPatchesd
from monai.transforms.utils import has_postive
from monai.utils import set_determinism
from tests.utils import TEST_NDARRAYS, assert_allclose

set_determinism(1234)

A = np.random.randint(64, 128, (3, 3, 2)).astype(np.uint8)
A3D = np.random.randint(64, 128, (3, 3, 2, 2)).astype(np.uint8)
B = np.ones_like(A[:1])
B3D = np.ones_like(A3D[:1])

MASK = np.pad(B, ((0, 0), (2, 2), (2, 2)), constant_values=0)
MASK3D = np.pad(B3D, ((0, 0), (2, 2), (2, 2), (2, 2)), constant_values=0)
IMAGE = np.pad(A, ((0, 0), (2, 2), (2, 2)), constant_values=255)
IMAGE3D = np.pad(A3D, ((0, 0), (2, 2), (2, 2), (2, 2)), constant_values=255)

IMAGE_255 = np.zeros_like(IMAGE) * 255
IMAGE_255_3D = np.zeros_like(IMAGE3D) * 255

image_patches_0 = IMAGE.repeat(4).reshape(3, 7, 6, 4).transpose(3, 0, 1, 2)
mask_patches_0 = MASK.repeat(4).reshape(1, 7, 6, 4).transpose(3, 0, 1, 2)

image_patches_1 = IMAGE3D.repeat(4).reshape(3, 7, 6, 6, 4).transpose(4, 0, 1, 2, 3)
mask_patches_1 = MASK3D.repeat(4).reshape(1, 7, 6, 6, 4).transpose(4, 0, 1, 2, 3)

image_patches_2 = np.array([IMAGE, IMAGE, IMAGE, IMAGE_255], dtype=np.uint8)
mask_patches_2 = np.array([MASK, MASK, MASK], dtype=np.uint8)

image_patches_3 = np.array([IMAGE3D, IMAGE3D, IMAGE3D, IMAGE_255_3D], dtype=np.uint8)
mask_patches_3 = np.array([MASK3D, MASK3D, MASK3D], dtype=np.uint8)

TEST_CASE_0 = [
    {"select_fn": has_postive, "threshold": "otsu", "hsv_threshold": None, "invert": False},
    image_patches_0,
    image_patches_0,
    mask_patches_0,
]
TEST_CASE_1 = [
    {"select_fn": has_postive, "threshold": "otsu", "hsv_threshold": None, "invert": False},
    image_patches_1,
    image_patches_1,
    mask_patches_1,
]
TEST_CASE_2 = [
    {"select_fn": has_postive, "threshold": "otsu", "hsv_threshold": None, "invert": False},
    image_patches_2,
    np.array([IMAGE, IMAGE, IMAGE], dtype=np.uint8),
    mask_patches_2,
]
TEST_CASE_3 = [
    {"select_fn": has_postive, "threshold": "otsu", "hsv_threshold": None, "invert": False},
    image_patches_3,
    np.array([IMAGE3D, IMAGE3D, IMAGE3D], dtype=np.uint8),
    mask_patches_3,
]

TEST_CASES = []
for p in TEST_NDARRAYS:
    TEST_CASES.append([p, *TEST_CASE_0])
    TEST_CASES.append([p, *TEST_CASE_1])
    TEST_CASES.append([p, *TEST_CASE_2])
    TEST_CASES.append([p, *TEST_CASE_3])


class TestGetForegroundPatchesd(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_grid_patch(self, in_type, input_parameters, image, expected_image, expected_mask):
        input_image = in_type(image)
        converter = GetForegroundPatchesd(keys="image", **input_parameters)
        input = {"image": input_image}
        ret = converter(input)

        self.assertEqual(len(ret["image"]), len(expected_image))
        self.assertEqual(len(ret["mask"]), len(expected_mask))
        for output_patch, expected_patch in zip(ret["image"], expected_image):
            assert_allclose(output_patch, expected_patch, type_test=False)

        for output_patch, expected_patch in zip(ret["mask"], expected_mask):
            assert_allclose(output_patch, expected_patch, type_test=False)


if __name__ == "__main__":
    unittest.main()
