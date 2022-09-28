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

import numpy as np
from parameterized import parameterized

from monai.apps.pathology.transforms.post.array import CalcualteInstanceSegmentationMap
from monai.utils import min_version, optional_import
from tests.utils import TEST_NDARRAYS, assert_allclose

_, has_skimage = optional_import("skimage", "0.19.3", min_version)
_, has_scipy = optional_import("scipy", "1.8.1", min_version)
_, has_cv2 = optional_import("cv2")

test_data_path = os.path.join(os.path.dirname(__file__), "testing_data", "hovernet_test_data.npz")
prediction = np.load(test_data_path)
params = {"threshold_overall": 0.4, "min_size": 10, "sigma": 0.4, "kernel_size": 21, "radius": 2}

TESTS = []
for p in TEST_NDARRAYS:
    seg_pred = prediction["seg_pred_act"]
    hover_map = prediction["hover_map"]
    expected = prediction["pred_instance"][None]

    TESTS.append([params, p, seg_pred, hover_map, expected])

ERROR_TESTS = []
for p in TEST_NDARRAYS:
    seg_pred1 = np.zeros([2, 64, 64])
    hover_map1 = np.zeros([2, 64, 64])

    seg_pred2 = np.zeros([1, 1, 64, 64])
    hover_map2 = np.zeros([2, 64, 64])

    seg_pred3 = np.zeros([1, 64, 64])
    hover_map3 = np.zeros([2, 1, 64, 64])

    ERROR_TESTS.append([params, p, seg_pred1, hover_map1])
    ERROR_TESTS.append([params, p, seg_pred2, hover_map2])
    ERROR_TESTS.append([params, p, seg_pred3, hover_map3])


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
@unittest.skipUnless(has_scipy, "Requires scipy library.")
@unittest.skipUnless(has_cv2, "OpenCV required.")
class TestCalcualteInstanceSegmentationMap(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, in_type, seg_pred, hover_map, expected):

        calculate_instance_seg = CalcualteInstanceSegmentationMap(**args)
        output = calculate_instance_seg(in_type(seg_pred), in_type(hover_map))

        self.assertTupleEqual(output.shape, expected.shape)
        assert_allclose(output, expected, type_test=False)

    @parameterized.expand(ERROR_TESTS)
    def test_value_error(self, args, in_type, seg_pred, hover_map):
        calculate_instance_seg = CalcualteInstanceSegmentationMap(**args)
        with self.assertRaises(ValueError):
            calculate_instance_seg(in_type(seg_pred), in_type(hover_map))


if __name__ == "__main__":
    unittest.main()
