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

from monai.apps.pathology.transforms.post.array import CalcualteInstanceSegmentationMap
from monai.transforms.intensity.array import ComputeHoVerMaps
from monai.utils import min_version, optional_import
from tests.utils import TEST_NDARRAYS

_, has_skimage = optional_import("skimage", "0.19.3", min_version)
_, has_scipy = optional_import("scipy", "1.8.1", min_version)


TESTS = []
for p in TEST_NDARRAYS:
    TEST_CASE_MASK = np.zeros((1, 10, 10), dtype="int16")
    TEST_CASE_MASK[:, 2:6, 2:6] = 1
    TEST_CASE_MASK[:, 7:10, 7:10] = 2
    mask = p(TEST_CASE_MASK)

    TEST_CASE_1 = np.zeros((1, 10, 10))
    TEST_CASE_1[:, 2:6, 2:6] = 0.7
    TEST_CASE_1[:, 7:10, 7:10] = 0.6

    probs_map_1 = p(TEST_CASE_1)
    expected_1 = (1, 10, 10)
    TESTS.append(
        [
            {
                "threshold_pred": 0.5,
                "threshold_overall": 0.4,
                "min_size": 4,
                "sigma": 0.4,
                "kernel_size": 3,
                "radius": 2,
            },
            p,
            probs_map_1,
            mask,
            expected_1,
        ]
    )


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
@unittest.skipUnless(has_scipy, "Requires scipy library.")
class TestGetInstanceLevelSegMap(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, in_type, probs_map, mask, expected):

        hover_map = in_type(ComputeHoVerMaps()(mask))
        getinstancelabel = CalcualteInstanceSegmentationMap(**args)
        output = getinstancelabel(probs_map, hover_map)

        # temporarily only test shape
        self.assertTupleEqual(output.shape, expected)


if __name__ == "__main__":
    unittest.main()
