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

import itertools
import unittest

import onnx
import onnxruntime as ort
import torch
from parameterized import parameterized

from monai.networks import convert_to_onnx
from monai.networks.nets import SegResNet, UNet

if "CUDAExecutionProvider" in ort.get_available_providers():
    ORT_PROVIDER_OPTIONS = [["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]]
else:
    ORT_PROVIDER_OPTIONS = [["CPUExecutionProvider"]]

if torch.cuda.is_available():
    TORCH_DEVICE_OPTIONS = ["cpu", "cuda"]
else:
    TORCH_DEVICE_OPTIONS = ["cpu"]
TESTS = list(itertools.product(TORCH_DEVICE_OPTIONS, [True, False], ORT_PROVIDER_OPTIONS))
TESTS_ORT = list(itertools.product(TORCH_DEVICE_OPTIONS, [True], ORT_PROVIDER_OPTIONS))


class TestConvertToOnnx(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_unet(self, device, use_ort, ort_provider):
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=0,
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

    @parameterized.expand(TESTS_ORT)
    def test_seg_res_net(self, device, use_ort, ort_provider):
        model = SegResNet(
            spatial_dims=3,
            init_filters=32,
            in_channels=1,
            out_channels=105,
            dropout_prob=0.2,
            act=('RELU', {'inplace': True}),
            norm=('GROUP', {'num_groups': 8}),
            norm_name='',
            num_groups=8,
            use_conv_final=True,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
        )
        onnx_model = convert_to_onnx(
            model=model,
            inputs=[torch.randn((1, 1, 24, 24, 24), requires_grad=False)],
            input_names=["x"],
            output_names=["y"],
            verify=True,
            device=device,
            use_ort=use_ort,
            ort_provider=ort_provider,
            use_trace=True,
            rtol=1e-3,
            atol=1e-4,
        )
        self.assertTrue(isinstance(onnx_model, onnx.ModelProto))


if __name__ == "__main__":
    unittest.main()
