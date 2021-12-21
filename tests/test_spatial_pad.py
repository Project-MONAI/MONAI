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
from typing import List

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import SpatialPad
from monai.utils.enums import NumpyPadMode, PytorchPadMode
from monai.utils.misc import set_determinism
from tests.utils import TEST_NDARRAYS

TESTS = []

MODES = []

# Test modes
NP_MODES: List = [
    "constant",
    "edge",
    # `reflect` mode is not supported in some PyTorch versions, skip the test
    # "reflect",
    "wrap",
]
MODES += NP_MODES
MODES += [NumpyPadMode(i) for i in NP_MODES]

PT_MODES: list = [
    "constant",
    "replicate",
    "circular",
    # `reflect` mode is not supported in some PyTorch versions, skip the test
    # "reflect",
]
MODES += PT_MODES
MODES += [PytorchPadMode(i) for i in PT_MODES]

for mode in MODES:
    TESTS.append([{"spatial_size": [3, 4], "method": "end", "mode": mode}, (1, 2, 3), (1, 3, 4)])

    TESTS.append([{"spatial_size": [15, 4, -1], "method": "symmetric", "mode": mode}, (3, 8, 8, 4), (3, 15, 8, 4)])


class TestSpatialPad(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(seed=0)

    def tearDown(self) -> None:
        set_determinism(None)

    @staticmethod
    def get_arr(shape):
        return np.random.randint(100, size=shape).astype(float)

    @parameterized.expand(TESTS)
    def test_pad_shape(self, input_param, input_shape, expected_shape):
        results_1 = []
        results_2 = []
        input_data = self.get_arr(input_shape)
        # check result is the same regardless of input type
        for p in TEST_NDARRAYS:
            padder = SpatialPad(**input_param)
            r1 = padder(p(input_data))
            r2 = padder(p(input_data), mode=input_param["mode"])
            results_1.append(r1.cpu() if isinstance(r1, torch.Tensor) else r1)
            results_2.append(r2.cpu() if isinstance(r2, torch.Tensor) else r2)
            for results in (results_1, results_2):
                np.testing.assert_allclose(results[-1].shape, expected_shape)
                if input_param["mode"] not in ("empty", NumpyPadMode.EMPTY):
                    torch.testing.assert_allclose(results[0], results[-1], atol=0, rtol=1e-5)

    def test_pad_kwargs(self):
        for p in TEST_NDARRAYS:
            input_data = p(np.zeros((3, 8, 4)))
            if isinstance(input_data, torch.Tensor):
                result = (
                    SpatialPad(spatial_size=[15, 8], method="end", mode="constant", value=2)(img=input_data)
                    .cpu()
                    .numpy()
                )
            else:
                result = SpatialPad(
                    spatial_size=[15, 8], method="end", mode="constant", constant_values=((0, 0), (1, 1), (2, 2))
                )(img=input_data)
                torch.testing.assert_allclose(result[:, 8:, :4], np.ones((3, 7, 4)), rtol=1e-7, atol=0)
            torch.testing.assert_allclose(result[:, :, 4:], np.ones((3, 15, 4)) + 1, rtol=1e-7, atol=0)


if __name__ == "__main__":
    unittest.main()
