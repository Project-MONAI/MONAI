# Copyright 2020 MONAI Consortium
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

from monai.transforms import SplitChanneld

TEST_CASE_1 = [
    {"keys": "pred", "output_postfixes": ["cls1", "cls2", "cls3"], "channel_dim": None},
    {"pred": torch.randint(0, 2, size=(4, 3, 3, 4))},
    (4, 1, 3, 4),
]

TEST_CASE_2 = [
    {"keys": "pred", "output_postfixes": ["cls1", "cls2", "cls3"], "channel_dim": 1},
    {"pred": torch.randint(0, 2, size=(4, 3, 3, 4))},
    (4, 1, 3, 4),
]

TEST_CASE_3 = [
    {"keys": "pred", "output_postfixes": ["cls1", "cls2", "cls3"], "channel_dim": None},
    {"pred": np.random.randint(2, size=(3, 3, 4))},
    (1, 3, 4),
]

TEST_CASE_4 = [
    {"keys": "pred", "output_postfixes": ["cls1", "cls2", "cls3"], "channel_dim": 0},
    {"pred": np.random.randint(2, size=(3, 3, 4))},
    (1, 3, 4),
]

TEST_CASE_5 = [
    {"keys": "pred", "output_postfixes": ["cls1", "cls2", "cls3", "cls4"], "channel_dim": 2},
    {"pred": np.random.randint(2, size=(3, 2, 4))},
    (3, 2, 1),
]

TEST_CASE_6 = [
    {"keys": "pred", "output_postfixes": ["cls1", "cls2", "cls3", "cls4"], "channel_dim": -1},
    {"pred": np.random.randint(2, size=(3, 2, 4))},
    (3, 2, 1),
]

TEST_CASE_7 = [
    {"keys": "pred", "channel_dim": 1},
    {"pred": np.random.randint(2, size=(3, 2, 4))},
    (3, 1, 4),
]


class TestSplitChanneld(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7])
    def test_shape(self, input_param, test_data, expected_shape):
        result = SplitChanneld(**input_param)(test_data)
        for k, v in result.items():
            if "_" in k:
                self.assertTupleEqual(v.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
