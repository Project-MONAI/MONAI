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

from monai.transforms.post.dictionary import GetInstancelabelledSegMapd
from monai.transforms.intensity.array import ComputeHoVerMaps
from tests.utils import TEST_NDARRAYS

compute_hover_maps = ComputeHoVerMaps()
TESTS = []
for p in TEST_NDARRAYS:
    TEST_CASE_MASK = np.zeros((1, 10, 10), dtype="int16")
    TEST_CASE_MASK[:, 2:6, 2:6] = 1
    TEST_CASE_MASK[:, 7:10, 7:10] = 2
    TEST_CASE_hover_map = compute_hover_maps(TEST_CASE_MASK)
    hover_map = p(TEST_CASE_hover_map)
    
    TEST_CASE_1 = np.zeros((1, 10, 10))
    TEST_CASE_1[:, 2:6, 2:6] = 0.7
    TEST_CASE_1[:, 7:10, 7:10] = 0.6

    probs_map_1 = p(TEST_CASE_1)
    expected_1 = (1, 10, 10)
    TESTS.append([{"keys": "image", "hover_key": "hover", "threshold_pred": 0.5, "threshold_overall": 0.4, "min_size": 4, "sigma": 0.4, "kernel_size": 3, "radius": 2}, probs_map_1, hover_map, expected_1])


class TestGetInstancelabelledSegMap(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, probs_map, hover_map, expected):
        data = {"image": probs_map, "hover": hover_map}
        output = GetInstancelabelledSegMapd(**args)(data)
        
        # temporarily only test shape
        self.assertTupleEqual(output["image"].shape, expected)


if __name__ == "__main__":
    unittest.main()
