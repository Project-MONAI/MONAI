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

from monai.transforms.post.array import GetInstancelabelledSegMap
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    TEST_CASE_1 = np.zeros((1, 16, 16), dtype="int16")
    TEST_CASE_1[..., 1:12,1:12] = 0.7
    TEST_CASE_1[..., 13:15,13:15] = 0.6

    TEST_CASE_hover_map_1 = np.array([
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0],
                            [0,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,0,0,0,0],
                            [0,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,0,0,0,0],
                            [0,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,-0.4,0,0,0,0],
                            [0,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0,0,0,0],
                            [0,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0,0,0,0],
                            [0,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0,0,0,0],
                            [0,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0,0,0,0],
                            [0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            ])
    TEST_CASE_hover_map_2 = np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])
    TEST_CASE_hover_map = np.stack([TEST_CASE_hover_map_1, TEST_CASE_hover_map_2], axis=0)
    print(TEST_CASE_hover_map.shape)

    probs_map_1 = p(TEST_CASE_1)
    hover_map_1 = p(TEST_CASE_hover_map)
    expected_1 = [[0.9, 66, 66], [0.7, 83, 33]]
    TESTS.append([{"threshold_pred": 0.5, "threshold_overall": 0.4, "min_size": 10, "sigma": 0.4, "kernel_size": 3, "radius": 2}, probs_map_1, hover_map_1, expected_1])



class TestGetInstancelabelledSegMap(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, probs_map, hover_map, expected):
        getinstancelabel = GetInstancelabelledSegMap(**args)
        output = getinstancelabel(probs_map, hover_map)

        # assert_allclose(output, expected)


if __name__ == "__main__":
    unittest.main()
