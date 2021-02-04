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

import torch
from parameterized import parameterized

from monai.networks.layers import grid_pull
from monai.utils import optional_import
from tests.utils import skip_if_no_cpp_extention

BType, has_b_type = optional_import("monai._C", name="BoundType")
PType, has_p_type = optional_import("monai._C", name="InterpolationType")


def make_grid(shape, dtype=None, device=None):
    ranges = [torch.arange(float(s), dtype=dtype, device=device) for s in shape]
    grid = torch.stack(torch.meshgrid(*ranges), dim=-1)
    return grid[None]


# 1D combinations of bounds/interpolations
bounds = set(BType.__members__.values()) if has_b_type else []
interps = set(PType.__members__.values()) if has_p_type else []
Expected_1D_BP_fwd = [torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])] * 56
assert len(bounds) * len(interps) == len(Expected_1D_BP_fwd)  # all combinations
TEST_1D_BP_fwd = []
for bound in bounds:
    for interp in interps:
        test_case = [
            {
                "input": torch.arange(10, dtype=torch.float).reshape((1, 1, 10)),
                "grid": make_grid((20,), dtype=torch.float) + 0.5,
                "interpolation": interp,
                "bound": bound,
            },
            Expected_1D_BP_fwd.pop(0),
        ]
        TEST_1D_BP_fwd.append(test_case)

@skip_if_no_cpp_extention
class TestGridPull(unittest.TestCase):
    @parameterized.expand(TEST_1D_BP_fwd)
    def test_grid_pull(self, input_param, expected_val):
        result = grid_pull(**input_param)
        print(input_param["interpolation"], input_param["bound"], result)
        # np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
