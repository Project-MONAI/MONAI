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

from monai.handlers import TrtHandler
from monai.networks import trt_compile
from monai.networks.nets import UNet, cell_sam_wrapper, vista3d132
from monai.utils import min_version, optional_import
from tests.utils import skip_if_no_cuda, skip_if_quick, skip_if_windows

trt, trt_imported = optional_import("tensorrt", "10.1.0", min_version)
polygraphy, polygraphy_imported = optional_import("polygraphy")
build_sam_vit_b, has_sam = optional_import("segment_anything.build_sam", name="build_sam_vit_b")

TEST_CASE_1 = ["fp32"]
TEST_CASE_2 = ["fp16"]


@skip_if_windows
@skip_if_no_cuda
@skip_if_quick
@unittest.skipUnless(trt_imported, "tensorrt is required")
@unittest.skipUnless(polygraphy_imported, "polygraphy is required")
class TestTRTCompile(unittest.TestCase):

    def setUp(self):
        self.gpu_device = torch.cuda.current_device()

    def tearDown(self):
        current_device = torch.cuda.current_device()
        if current_device != self.gpu_device:
            torch.cuda.set_device(self.gpu_device)

    def test_handler(self):
        from ignite.engine import Engine

        net1 = torch.nn.Sequential(*[torch.nn.PReLU(), torch.nn.PReLU()])
        data1 = net1.state_dict()
        data1["0.weight"] = torch.tensor([0.1])
        data1["1.weight"] = torch.tensor([0.2])
        net1.load_state_dict(data1)
        net1.cuda()

        with tempfile.TemporaryDirectory() as tempdir:
            engine = Engine(lambda e, b: None)
            args = {"method": "torch_trt"}
            TrtHandler(net1, tempdir + "/trt_handler", args=args).attach(engine)
            engine.run([0] * 8, max_epochs=1)
            self.assertIsNotNone(net1._trt_compiler)
            self.assertIsNone(net1._trt_compiler.engine)
            net1.forward(torch.tensor([[0.0, 1.0], [1.0, 2.0]], device="cuda"))
            self.assertIsNotNone(net1._trt_compiler.engine)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_unet_value(self, precision):
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
            input_example = torch.randn(2, 1, 96, 96, 96).cuda()
            output_example = model(input_example)
            args: dict = {"builder_optimization_level": 1}
            trt_compile(
                model,
                f"{tmpdir}/test_unet_trt_compile",
                args={"precision": precision, "build_args": args, "dynamic_batchsize": [1, 4, 8]},
            )
            self.assertIsNone(model._trt_compiler.engine)
            trt_output = model(input_example)
            # Check that lazy TRT build succeeded
            self.assertIsNotNone(model._trt_compiler.engine)
            torch.testing.assert_close(trt_output, output_example, rtol=0.01, atol=0.01)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    @unittest.skipUnless(has_sam, "Requires SAM installation")
    def test_cell_sam_wrapper_value(self, precision):
        model = cell_sam_wrapper.CellSamWrapper(checkpoint=None).to("cuda")
        with torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
            model.eval()
            input_example = torch.randn(1, 3, 128, 128).to("cuda")
            output_example = model(input_example)
            trt_compile(
                model,
                f"{tmpdir}/test_cell_sam_wrapper_trt_compile",
                args={"precision": precision, "dynamic_batchsize": [1, 1, 1]},
            )
            self.assertIsNone(model._trt_compiler.engine)
            trt_output = model(input_example)
            # Check that lazy TRT build succeeded
            self.assertIsNotNone(model._trt_compiler.engine)
            torch.testing.assert_close(trt_output, output_example, rtol=0.01, atol=0.01)

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_vista3d(self, precision):
        model = vista3d132(in_channels=1).to("cuda")
        with torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
            model.eval()
            input_example = torch.randn(1, 1, 64, 64, 64).to("cuda")
            output_example = model(input_example)
            model = trt_compile(
                model,
                f"{tmpdir}/test_vista3d_trt_compile",
                args={"precision": precision, "dynamic_batchsize": [1, 1, 1]},
                submodule=["image_encoder.encoder", "class_head"],
            )
            self.assertIsNotNone(model.image_encoder.encoder._trt_compiler)
            self.assertIsNotNone(model.class_head._trt_compiler)
            trt_output = model.forward(input_example)
            # Check that lazy TRT build succeeded
            # TODO: set up input_example in such a way that image_encoder.encoder and class_head are called
            # and uncomment the asserts below
            # self.assertIsNotNone(model.image_encoder.encoder._trt_compiler.engine)
            # self.assertIsNotNone(model.class_head._trt_compiler.engine)
            torch.testing.assert_close(trt_output, output_example, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
