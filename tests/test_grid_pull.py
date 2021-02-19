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

from monai.networks.layers import grid_pull
from monai.utils import optional_import
from tests.testing_data.cpp_resample_answers import Expected_1D_GP_fwd
from tests.utils import skip_if_no_cpp_extension

BType, has_b_type = optional_import("monai._C", name="BoundType")
PType, has_p_type = optional_import("monai._C", name="InterpolationType")


def make_grid(shape, dtype=None, device=None):
    ranges = [torch.arange(float(s), dtype=dtype, device=device, requires_grad=True) for s in shape]
    grid = torch.stack(torch.meshgrid(*ranges), dim=-1)
    return grid[None]


# 1D combinations of bounds/interpolations
bounds = set(BType.__members__.values()) if has_b_type else []
interps = set(PType.__members__.values()) if has_p_type else []
device = "cuda" if torch.cuda.is_available() else "cpu"
TEST_1D_GP_fwd = []
for bound in bounds:
    for interp in interps:
        if not Expected_1D_GP_fwd:
            break  # skip if the testing data are unavailable
        test_case = [
            {
                "input": torch.arange(10, dtype=torch.float, requires_grad=True, device=device).reshape((1, 1, 10)),
                "grid": make_grid((20,), dtype=torch.float, device=device) + 0.5,
                "interpolation": interp,
                "bound": bound,
            },
            {
                "val": torch.tensor([[Expected_1D_GP_fwd.pop(0)]]),
                "device": device,
            },
        ]
        TEST_1D_GP_fwd.append(test_case)


@skip_if_no_cpp_extension
class TestGridPull(unittest.TestCase):
    @parameterized.expand(TEST_1D_GP_fwd, skip_on_empty=True)
    def test_grid_pull(self, input_param, expected):
        result = grid_pull(**input_param)
        self.assertTrue("{}".format(result.device).startswith(expected["device"]))
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected["val"].cpu().numpy(), rtol=1e-4, atol=1e-4)

    @parameterized.expand(TEST_1D_GP_fwd, skip_on_empty=True)
    def test_grid_pull_grad(self, input_param, expected):
        result = grid_pull(**input_param)
        input_param["input"].retain_grad()
        input_param["grid"].retain_grad()
        result.sum().backward()
        print("--" * 15)
        print(input_param["interpolation"], input_param["bound"])
        print(input_param["input"].grad)
        print(input_param["grid"].grad)


if __name__ == "__main__":
    unittest.main()
