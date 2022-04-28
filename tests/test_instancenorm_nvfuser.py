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

import torch
from parameterized import parameterized

from monai.networks.layers.nvfuser import InstanceNorm3dNVFuser
from monai.utils import optional_import
from tests.utils import SkipIfBeforePyTorchVersion, skip_if_no_cuda, skip_if_quick, skip_if_windows

_, has_nvfuser = optional_import("instance_norm_nvfuser_cuda")


TEST_CASES = []
input_shape = (1, 3, 64, 64, 64)
for eps in [1e-4, 1e-5]:
    for momentum in [0.1, 0.01]:
        for affine in [True, False]:
            test_case = [
                {
                    "num_features": input_shape[1],
                    "eps": eps,
                    "momentum": momentum,
                    "affine": affine,
                    "device": "cuda",
                },
                input_shape,
            ]
            TEST_CASES.append(test_case)


@skip_if_no_cuda
@skip_if_windows
@skip_if_quick
@SkipIfBeforePyTorchVersion((1, 10))
@unittest.skipUnless(has_nvfuser, "`instance_norm_nvfuser_cuda` is necessary for `InstanceNorm3dNVFuser`.")
class TestInstanceNorm3dNVFuser(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_layer_consistency(self, input_param, input_shape):
        input_tensor = torch.randn(input_shape).to("cuda")
        in_layer = torch.nn.InstanceNorm3d(**input_param)
        in_3dnvfuser_layer = InstanceNorm3dNVFuser(**input_param)
        out_in = in_layer(input_tensor)
        out_3dnvfuser = in_3dnvfuser_layer(input_tensor)

        torch.testing.assert_close(out_in, out_3dnvfuser)


if __name__ == "__main__":
    unittest.main()
