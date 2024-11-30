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

from __future__ import annotations

import unittest
from unittest import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import RandTorchIO
from monai.utils import optional_import, set_determinism

_, has_torchio = optional_import("torchio")

TEST_DIMS = [3, 128, 160, 160]
TESTS = [
    [{"name": "RandomAffine"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomElasticDeformation"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomAnisotropy"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomMotion"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomGhosting"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomSpike"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomBiasField"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomBlur"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomNoise"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomSwap"}, torch.rand(TEST_DIMS)],
    [{"name": "RandomGamma"}, torch.rand(TEST_DIMS)],
]


@skipUnless(has_torchio, "Requires torchio")
class TestRandTorchIO(unittest.TestCase):

    @parameterized.expand(TESTS)
    def test_value(self, input_param, input_data):
        set_determinism(seed=0)
        result = RandTorchIO(**input_param)(input_data)
        self.assertIsNotNone(result)
        self.assertFalse(np.array_equal(result.numpy(), input_data.numpy()), f"{input_param} failed")


if __name__ == "__main__":
    unittest.main()
