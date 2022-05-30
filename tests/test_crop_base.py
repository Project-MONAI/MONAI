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

from parameterized import parameterized

from monai.transforms import SpatialCrop
from monai.transforms.croppad.array import CropBase

TEST_ERRORS = [
    [{k: None for k in ("roi_slices", "roi_center", "roi_size", "roi_start", "roi_end")}],
    [{k: None for k in ("roi_slices", "roi_center")}],
    [{k: None for k in ("roi_slices", "roi_center", "roi_size")}],
    [{k: None for k in ("roi_size", "roi_end")}],
    [{k: None for k in ("roi_end",)}],
    [{k: None for k in ("roi_center",)}],
]

TESTS = [
    # [
    #     {"roi_slices": [slice(None), slice(None), slice(None)]},
    #     [slice(None), slice(None), slice(None)],
    # ],
    # [
    #     {"roi_start": (0, 0, 0), "roi_end": (10, -1, 2)},
    #     [slice(0, 10, None), slice(0, -1, None), slice(0, 2, None)],
    # ],
    [
        {"roi_center": (10, 6, 13), "roi_size": (3, 5, -1)},
        [slice(0, 10, None), slice(0, -1, None), slice(0, 2, None)],
    ],
]



class TestCropBase(unittest.TestCase):
    @parameterized.expand(TEST_ERRORS)
    def test_error(self, input_param):
        with self.assertRaises(ValueError):
            SpatialCrop(**input_param)

    def test_slice_calculation(self, roi_params, expected_slices):
        slices = CropBase.calculate_slices(**roi_params)
        self.assertEqual(slices, expected_slices)




if __name__ == "__main__":
    # unittest.main()
    a = TestCropBase()
    for t in TESTS:
        a.test_slice_calculation(*t)
