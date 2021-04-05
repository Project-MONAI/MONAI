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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms.post.dictionary import ProbNMSD

probs_map_1 = np.random.rand(100, 100).clip(0, 0.5)
TEST_CASES_2D_1 = [{"spatial_dims": 2, "prob_threshold": 0.5, "box_size": 10}, {"prob_map": probs_map_1}, []]

probs_map_2 = np.random.rand(100, 100).clip(0, 0.5)
probs_map_2[33, 33] = 0.7
probs_map_2[66, 66] = 0.9
expected_2 = [[0.9, 66, 66], [0.7, 33, 33]]
TEST_CASES_2D_2 = [
    {"spatial_dims": 2, "prob_threshold": 0.5, "box_size": [10, 10]},
    {"prob_map": probs_map_2},
    expected_2,
]

probs_map_3 = np.random.rand(100, 100).clip(0, 0.5)
probs_map_3[56, 58] = 0.7
probs_map_3[60, 66] = 0.8
probs_map_3[66, 66] = 0.9
expected_3 = [[0.9, 66, 66], [0.8, 60, 66]]
TEST_CASES_2D_3 = [
    {"spatial_dims": 2, "prob_threshold": 0.5, "box_size": (10, 20)},
    {"prob_map": probs_map_3},
    expected_3,
]

probs_map_4 = np.random.rand(100, 100).clip(0, 0.5)
probs_map_4[33, 33] = 0.7
probs_map_4[66, 66] = 0.9
expected_4 = [[0.9, 66, 66]]
TEST_CASES_2D_4 = [
    {"spatial_dims": 2, "prob_threshold": 0.8, "box_size": 10},
    {"prob_map": probs_map_4},
    expected_4,
]

probs_map_5 = np.random.rand(100, 100).clip(0, 0.5)
TEST_CASES_2D_5 = [{"spatial_dims": 2, "prob_threshold": 0.5, "sigma": 0.1}, {"prob_map": probs_map_5}, []]

probs_map_6 = torch.as_tensor(np.random.rand(100, 100).clip(0, 0.5))
TEST_CASES_2D_6 = [{"spatial_dims": 2, "prob_threshold": 0.5, "sigma": 0.1}, {"prob_map": probs_map_6}, []]

probs_map_7 = torch.as_tensor(np.random.rand(100, 100).clip(0, 0.5))
probs_map_7[33, 33] = 0.7
probs_map_7[66, 66] = 0.9
if torch.cuda.is_available():
    probs_map_7 = probs_map_7.cuda()
expected_7 = [[0.9, 66, 66], [0.7, 33, 33]]
TEST_CASES_2D_7 = [
    {"spatial_dims": 2, "prob_threshold": 0.5, "sigma": 0.1},
    {"prob_map": probs_map_7},
    expected_7,
]

probs_map_3d = torch.rand([50, 50, 50]).uniform_(0, 0.5)
probs_map_3d[25, 25, 25] = 0.7
probs_map_3d[45, 45, 45] = 0.9
expected_3d = [[0.9, 45, 45, 45], [0.7, 25, 25, 25]]
TEST_CASES_3D = [
    {"spatial_dims": 3, "prob_threshold": 0.5, "box_size": (10, 10, 10)},
    {"prob_map": probs_map_3d},
    expected_3d,
]


class TestProbNMS(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASES_2D_1,
            TEST_CASES_2D_2,
            TEST_CASES_2D_3,
            TEST_CASES_2D_4,
            TEST_CASES_2D_5,
            TEST_CASES_2D_6,
            TEST_CASES_2D_7,
            TEST_CASES_3D,
        ]
    )
    def test_output(self, class_args, probs_map, expected):
        nms = ProbNMSD(keys="prob_map", **class_args)
        output = nms(probs_map)
        np.testing.assert_allclose(output["prob_map"], expected)


if __name__ == "__main__":
    unittest.main()
