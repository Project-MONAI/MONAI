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
from copy import deepcopy

from parameterized import parameterized

from monai.transforms import SpatialCrop
from monai.transforms.croppad.array import CropBase
from tests.utils import TEST_NDARRAYS

TEST_ERRORS = [
    [{k: None for k in ("roi_slices", "roi_center", "roi_size", "roi_start", "roi_end")}],
    [{k: None for k in ("roi_slices", "roi_center")}],
    [{k: None for k in ("roi_slices", "roi_center", "roi_size")}],
    [{k: None for k in ("roi_size", "roi_end")}],
    [{k: None for k in ("roi_end",)}],
    [{k: None for k in ("roi_center",)}],
]

TESTS = [
    [  # slices given, should be same returned
        {"roi_slices": [slice(None), slice(None), slice(None)]},
        [slice(None), slice(None), slice(None)],
    ],
    [  # slices given, should be same returned
        {"roi_slices": [slice(-1, 3), slice(-3, 6), slice(None)]},
        [slice(-1, 3), slice(-3, 6), slice(None)],
    ],
    [  # slices are just start and end
        {"roi_start": (0,), "roi_end": (10,)},
        [slice(0, 10, None)],
    ],
    [  # slices are just start and end
        {"roi_start": (0, 0, 0), "roi_end": (10, -1, 2)},
        [slice(0, 10, None), slice(0, -1, None), slice(0, 2, None)],
    ],
    [  # start/end = center -/+ half of roi size. when size is -ve, no cropping, so slice(None) returned.
        {"roi_center": (10, ), "roi_size": (3, )},
        [slice(9, 11, None)],
    ],
    [  # start/end = center -/+ half of roi size. when size is -ve, no cropping, so slice(None) returned.
        {"roi_center": (10, 6, 13), "roi_size": (3, 4, -1)},
        [slice(9, 11, None), slice(4, 7, None), slice(None)],
    ],
    [  # start and end. when center - size // 2 is neg, min set to 0
        {"roi_center": (2, 6), "roi_size": (9, -1)},
        [slice(0, 6, None), slice(None)],
    ],
]


class TestCropBase(unittest.TestCase):
    @parameterized.expand(TEST_ERRORS)
    def test_error(self, input_param):
        with self.assertRaises(ValueError):
            SpatialCrop(**input_param)

    @parameterized.expand(TESTS)
    def test_slice_calculation(self, roi_params, expected_slices):
        # input parameters, such as roi_start can be numpy, torch, list etc.
        for param_type in TEST_NDARRAYS + (None,):
            with self.subTest(param_type=param_type):
                roi_params_mod = deepcopy(roi_params)
                if param_type is not None:
                    for k in ("roi_start", "roi_end", "roi_center", "roi_size"):
                        if k in roi_params:
                            roi_params_mod[k] = param_type(roi_params[k])
        slices = CropBase.calculate_slices(**roi_params)
        self.assertEqual(slices, expected_slices)


if __name__ == "__main__":
    unittest.main()