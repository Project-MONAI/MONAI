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

import numpy as np
import torch
from parameterized import parameterized

from monai.inferers import SlidingWindowInferer, SlidingWindowInfererExport
from monai.utils import optional_import

_, _has_onnx = optional_import("onnx")
_, _has_onnxruntime = optional_import("onnxruntime")

# (image shape, roi_size, sw_batch_size, overlap, mode)
TEST_CASES = [
    [(1, 1, 32, 40), (16, 16), 2, 0.25, "constant"],  # 2D
    [(1, 1, 32, 40), (16, 16), 2, 0.25, "gaussian"],  # 2D gaussian blending
    [(2, 1, 32, 40), (16, 16), 1, 0.25, "constant"],  # 2D batch of images, one window at a time
    [(3, 1, 32, 40), (16, 16), 4, 0.5, "gaussian"],  # 2D batch, large overlap
    [(1, 1, 24, 28, 20), (16, 16, 16), 2, 0.25, "constant"],  # 3D
    [(1, 1, 20, 22, 18), (12, 12, 12), 3, 0.5, "gaussian"],  # 3D gaussian blending
    [(1, 1, 32, 40), (48, 48), 2, 0.25, "constant"],  # roi larger than image (padding)
    [(1, 1, 32, 36), (48, 40), 2, 0.25, "gaussian"],  # roi larger than image, gaussian
    [(1, 1, 33), (16,), 2, 0.25, "constant"],  # 1D
]


class _TinyNet(torch.nn.Module):
    """A small convolutional network preserving the spatial size of its input."""

    def __init__(self, spatial_dims: int, in_channels: int = 1, out_channels: int = 2):
        super().__init__()
        conv = getattr(torch.nn, f"Conv{spatial_dims}d")
        self.conv = conv(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # type: ignore


def _make_case(image_shape, roi_size, sw_batch_size, overlap, mode):
    spatial_dims = len(image_shape) - 2
    torch.manual_seed(123)
    net = _TinyNet(spatial_dims, in_channels=image_shape[1]).eval()
    inputs = torch.randn(*image_shape)
    inferer = SlidingWindowInferer(
        roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap, mode=mode, progress=False
    )
    with torch.no_grad():
        expected = inferer(inputs, net)
    module = SlidingWindowInfererExport(
        net, roi_size=roi_size, spatial_size=image_shape[2:], sw_batch_size=sw_batch_size, overlap=overlap, mode=mode
    ).eval()
    return module, inputs, expected


class TestSlidingWindowInfererExport(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_eager_matches_sliding_window_inferer(self, image_shape, roi_size, sw_batch_size, overlap, mode):
        module, inputs, expected = _make_case(image_shape, roi_size, sw_batch_size, overlap, mode)
        with torch.no_grad():
            result = module(inputs)
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)

    @parameterized.expand(TEST_CASES)
    def test_onnx_export_matches_sliding_window_inferer(self, image_shape, roi_size, sw_batch_size, overlap, mode):
        if not (_has_onnx and _has_onnxruntime):
            self.skipTest("optional modules not present: onnx and/or onnxruntime")
        import onnxruntime as ort

        module, inputs, expected = _make_case(image_shape, roi_size, sw_batch_size, overlap, mode)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                module,
                (inputs,),
                f.name,
                input_names=["input"],
                output_names=["output"],
                opset_version=18,
                dynamo=False,
            )
            session = ort.InferenceSession(f.name, providers=["CPUExecutionProvider"])
            (onnx_result,) = session.run(None, {"input": inputs.numpy()})
        self.assertEqual(onnx_result.shape, expected.shape)
        np.testing.assert_allclose(onnx_result, expected.numpy(), rtol=1e-5, atol=1e-5)

    def test_spatial_size_mismatch_raises(self):
        net = _TinyNet(2).eval()
        module = SlidingWindowInfererExport(net, roi_size=(16, 16), spatial_size=(32, 32)).eval()
        with self.assertRaisesRegex(ValueError, "spatial size"):
            module(torch.randn(1, 1, 24, 24))

    def test_multi_output_network_raises(self):
        class _TwoOutputs(torch.nn.Module):
            def forward(self, x):
                return x, x

        module = SlidingWindowInfererExport(_TwoOutputs(), roi_size=(16, 16), spatial_size=(32, 32)).eval()
        with self.assertRaisesRegex(TypeError, "single tensor"):
            module(torch.randn(1, 1, 32, 32))

    def test_invalid_overlap_raises(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            SlidingWindowInfererExport(_TinyNet(2), roi_size=(16, 16), spatial_size=(32, 32), overlap=1.0)


if __name__ == "__main__":
    unittest.main()
