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

from monai.transforms import MeanEnsembled

TEST_CASE_1 = [
    {"keys": ["pred0", "pred1"], "output_key": "output", "weights": None},
    {"pred0": torch.ones(2, 2, 2), "pred1": torch.ones(2, 2, 2) + 2},
    torch.ones(2, 2, 2) + 1,
]

TEST_CASE_2 = [
    {"keys": "output", "weights": None},
    {"output": torch.stack([torch.ones(2, 2, 2), torch.ones(2, 2, 2) + 2])},
    torch.ones(2, 2, 2) + 1,
]

TEST_CASE_3 = [
    {"keys": ["pred0", "pred1"], "output_key": "output", "weights": [1, 3]},
    {"pred0": torch.ones(2, 2, 2, 2), "pred1": torch.ones(2, 2, 2, 2) + 2},
    torch.ones(2, 2, 2, 2) * 2.5,
]

TEST_CASE_4 = [
    {"keys": ["pred0", "pred1"], "output_key": "output", "weights": [[1, 3], [3, 1]]},
    {"pred0": torch.ones(2, 2, 2), "pred1": torch.ones(2, 2, 2) + 2},
    torch.ones(2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(2, 1, 1),
]

TEST_CASE_5 = [
    {"keys": ["pred0", "pred1"], "output_key": "output", "weights": np.array([[[1, 3]], [[3, 1]]])},
    {"pred0": torch.ones(2, 2, 2, 2), "pred1": torch.ones(2, 2, 2, 2) + 2},
    torch.ones(2, 2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(1, 2, 1, 1),
]

TEST_CASE_6 = [
    {"keys": ["pred0", "pred1"], "output_key": "output", "weights": torch.tensor([[[1, 3]], [[3, 1]]])},
    {"pred0": torch.ones(2, 2, 2, 2), "pred1": torch.ones(2, 2, 2, 2) + 2},
    torch.ones(2, 2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(1, 2, 1, 1),
]


class TestMeanEnsembled(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_value(self, input_param, data, expected_value):
        result = MeanEnsembled(**input_param)(data)
        torch.testing.assert_allclose(result["output"], expected_value)

    def test_cuda_value(self):
        img = torch.stack([torch.ones(2, 2, 2, 2), torch.ones(2, 2, 2, 2) + 2])
        expected_value = torch.ones(2, 2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(1, 2, 1, 1)
        if torch.cuda.is_available():
            img = img.to(torch.device("cuda:0"))
            expected_value = expected_value.to(torch.device("cuda:0"))
        result = MeanEnsembled(keys="output", weights=torch.tensor([[[1, 3]], [[3, 1]]]))({"output": img})
        torch.testing.assert_allclose(result["output"], expected_value)


if __name__ == "__main__":
    unittest.main()
