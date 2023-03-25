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
import itertools
from parameterized import parameterized

import onnx
import torch

from monai.networks import convert_to_onnx
from monai.networks.nets import UNet


TESTS = list(itertools.product(["cpu", "cuda"], [True, False], ["CUDAExecutionProvider", "CPUExecutionProvider"]))

class TestConvertToOnnx(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, device, use_ort=False, ort_provider=None):
        model = UNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((16, 1, 32, 32), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            ort_provider=ort_provider,
            rtol=1e-3,
            atol=1e-4,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))


if __name__ == "__main__":
    unittest.main()
