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

import tempfile
import unittest

import torch
from ignite.engine import Engine, Events
from parameterized import parameterized

from monai.handlers import TrtHandler
from monai.networks.nets import UNet
from monai.utils import optional_import
from tests.utils import skip_if_no_cuda, skip_if_quick, skip_if_windows

trt_compile, has_trt = optional_import(
    "monai.networks", name="trt_compile", descriptor="TRT compile is not available - check your installation!"
)

TEST_CASE_1 = ["fp32"]
TEST_CASE_2 = ["fp16"]


@skip_if_windows
@skip_if_no_cuda
@skip_if_quick
class TestTRTCompile(unittest.TestCase):

    def setUp(self):
        self.gpu_device = torch.cuda.current_device()

    def tearDown(self):
        current_device = torch.cuda.current_device()
        if current_device != self.gpu_device:
            torch.cuda.set_device(self.gpu_device)

    @unittest.skipUnless(has_trt, "TensorRT compile wrapper is required for convert!")
    def test_handler(self):
        net1 = torch.nn.Sequential(*[torch.nn.PReLU(), torch.nn.PReLU()])
        data1 = net1.state_dict()
        data1["0.weight"] = torch.tensor([0.1])
        data1["1.weight"] = torch.tensor([0.2])
        net1.load_state_dict(data1)
        net1.cuda()

        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            TrtHandler(net1, tempdir + "/trt_handler").attach(engine)
            engine.run([0] * 8, max_epochs=1)
            self.assertIsNotNone(net1._trt_compiler)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @unittest.skipUnless(has_trt, "TensorRT compile wrapper is required for convert!")
    def test_value(self, precision):
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(2, 2, 4, 8, 4),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        ).cuda()
        with torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
            model.eval()
            input_example = torch.randn(1, 1, 96, 96, 96).cuda()
            output_example = model(input_example)
            args: dict = {"builder_optimization_level": 1}
            trt_compile(
                model,
                f"{tmpdir}/test_trt_compile",
                args={"precision": precision, "build_args": args, "dynamic_batchsize": [1, 4, 8]},
            )
            self.assertIsNone(model._trt_compiler.engine)
            trt_output = model(input_example)
            # Check that lazy TRT build succeeded
            self.assertIsNotNone(model._trt_compiler.engine)
            torch.testing.assert_close(trt_output, output_example, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
