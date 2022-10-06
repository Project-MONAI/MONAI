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

from monai.apps.pathology.transforms.post.array import CalculateInstanceSegmentationMap
from monai.utils import min_version, optional_import
from tests.utils import TEST_NDARRAYS

_, has_skimage = optional_import("skimage", "0.19.3", min_version)
_, has_scipy = optional_import("scipy", "1.8.1", min_version)
_, has_cv2 = optional_import("cv2")

test_data_path = os.path.join(os.path.dirname(__file__), "testing_data", "hovernet_test_data.npz")
prediction = np.load(test_data_path)


np.random.RandomState(123)

TESTS = []
params = {"connectivity": 1}
for p in TEST_NDARRAYS:
    image = p(np.random.rand(1, 10, 10))
    mask = p((np.random.rand(1, 10, 10)>0.5).astype(np.uint8))
    marker = p((np.random.rand(1, 10, 10)>0.5).astype(np.uint8))

    TESTS.append([params, image, mask, marker, (1, 10, 10)])

ERROR_TESTS = []
for p in TEST_NDARRAYS:
    image = p(np.random.rand(3, 10, 10))
    mask = p((np.random.rand(2, 10, 10)>0.5).astype(np.uint8))
    marker = p((np.random.rand(2, 10, 10)>0.5).astype(np.uint8))

    ERROR_TESTS.append([params, image, mask, marker])


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
@unittest.skipUnless(has_scipy, "Requires scipy library.")
@unittest.skipUnless(has_cv2, "OpenCV required.")
class TestCalculateInstanceSegmentationMap(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, image, mask, markers, expected_shape):
        calculate_instance_seg = CalculateInstanceSegmentationMap(**args)
        output = calculate_instance_seg(image, mask, markers)

        self.assertTupleEqual(output.shape, expected_shape)

    @parameterized.expand(ERROR_TESTS)
    def test_value_error(self, args, image, mask, markers):
        calculate_instance_seg = CalculateInstanceSegmentationMap(**args)
        with self.assertRaises(ValueError):
            calculate_instance_seg(image, mask, markers)


if __name__ == "__main__":
    unittest.main()
