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
from parameterized import parameterized

from monai.networks.nets import UNet
from monai.utils import optional_import
from tests.utils import skip_if_no_cuda, skip_if_quick, skip_if_windows

TRTWrapper, has_trtwrapper = optional_import(
    "monai.utils", name="TRTWrapper", descriptor="TRT wrapper is not available - check your installation!"
)

TEST_CASE_1 = ["fp32"]
TEST_CASE_2 = ["fp16"]


@skip_if_windows
@skip_if_no_cuda
@skip_if_quick
class TestConvertToTRT(unittest.TestCase):

    def setUp(self):
        self.gpu_device = torch.cuda.current_device()

    def tearDown(self):
        current_device = torch.cuda.current_device()
        if current_device != self.gpu_device:
            torch.cuda.set_device(self.gpu_device)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @unittest.skipUnless(has_trtwrapper, "TensorRT wrapper is required for convert!")
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
        with torch.no_grad(), tempfile.TemporaryDirectory() as _:
            model.eval()
            input_example = torch.randn(1, 1, 96, 96, 96).cuda()
            output_example = model(input_example)
            args: dict = {"tf32": True}
            if precision == "fp16":
                args["fp16"] = True
                args["precision_constraints"] = "obey"

            trt_wrapper = TRTWrapper("test_wrapper", model, input_names=["x"])
            trt_wrapper.build_and_save(input_example, **args, builder_optimization_level=1)
            trt_output = trt_wrapper(x=input_example)
            torch.testing.assert_close(trt_output, output_example, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
