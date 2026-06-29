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

import math
import unittest
from unittest import skipUnless

import torch
import torch.nn as nn
from parameterized import parameterized

from monai.networks.blocks.hyena import (
    DepthwiseFFTConv2d,
    DepthwiseFFTConv3d,
    HyenaMixer,
    HyenaTransformerBlock,
    is_nvsubquadratic_available,
)

HAS_NVSUBQ = is_nvsubquadratic_available()
HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# DepthwiseFFTConv{2,3}d — no nvsubquadratic dependency
# ---------------------------------------------------------------------------


class TestDepthwiseFFTConvShape(unittest.TestCase):
    """The FFT conv must preserve spatial dimensions for any depthwise config."""

    @parameterized.expand(
        [
            ("3d_k3_d16", (2, 8, 16, 16, 16), 3, 3),
            ("3d_k1_d10", (1, 16, 10, 10, 10), 1, 3),
            ("3d_k5_d12", (1, 8, 12, 12, 12), 5, 3),
            ("2d_k3_d32", (2, 8, 32, 32), 3, 2),
        ]
    )
    def test_output_shape(self, _name, input_shape, kernel_size, spatial_dims):
        channels = input_shape[1]
        cls = DepthwiseFFTConv3d if spatial_dims == 3 else DepthwiseFFTConv2d
        conv = cls(channels, channels, kernel_size=kernel_size, groups=channels, padding=kernel_size // 2)
        x = torch.randn(*input_shape)
        self.assertEqual(conv(x).shape, x.shape)


class TestDepthwiseFFTConvNumerics(unittest.TestCase):
    """FFT conv must match the equivalent ``nn.Conv{2,3}d`` numerically."""

    @parameterized.expand([("d8_s12", 8, 12), ("d16_s8", 16, 8), ("d32_s6", 32, 6)])
    def test_matches_conv3d(self, _name, channels, spatial):
        ref = nn.Conv3d(channels, channels, kernel_size=3, groups=channels, padding=1, bias=False)
        fft = DepthwiseFFTConv3d(channels, channels, kernel_size=3, groups=channels, padding=1)
        with torch.no_grad():
            fft.weight.copy_(ref.weight)
        x = torch.randn(2, channels, spatial, spatial, spatial)
        with torch.no_grad():
            torch.testing.assert_close(fft(x), ref(x), atol=1e-4, rtol=1e-4)

    def test_matches_conv2d(self):
        channels, spatial = 8, 16
        ref = nn.Conv2d(channels, channels, kernel_size=3, groups=channels, padding=1, bias=False)
        fft = DepthwiseFFTConv2d(channels, channels, kernel_size=3, groups=channels, padding=1)
        with torch.no_grad():
            fft.weight.copy_(ref.weight)
        x = torch.randn(2, channels, spatial, spatial)
        with torch.no_grad():
            torch.testing.assert_close(fft(x), ref(x), atol=1e-4, rtol=1e-4)


class TestDepthwiseFFTConvDtype(unittest.TestCase):
    """Output dtype must match input dtype (AMP transparency)."""

    @parameterized.expand([("fp16", torch.float16), ("bf16", torch.bfloat16)])
    def test_amp_dtype_preserved(self, _name, dtype):
        conv = DepthwiseFFTConv3d(8, 8, kernel_size=3, groups=8, padding=1)
        x = torch.randn(1, 8, 8, 8, 8, dtype=dtype)
        out = conv(x)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(out.shape, x.shape)

    def test_float32_preserved(self):
        conv = DepthwiseFFTConv3d(8, 8, kernel_size=3, groups=8, padding=1)
        x = torch.randn(1, 8, 8, 8, 8)
        self.assertEqual(conv(x).dtype, torch.float32)


class TestDepthwiseFFTConvGradients(unittest.TestCase):
    """Backward pass must produce gradients on both input and weight."""

    def test_gradients_flow_3d(self):
        conv = DepthwiseFFTConv3d(8, 8, kernel_size=3, groups=8, padding=1)
        x = torch.randn(1, 8, 10, 10, 10, requires_grad=True)
        conv(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)

    def test_gradients_flow_2d(self):
        conv = DepthwiseFFTConv2d(8, 8, kernel_size=3, groups=8, padding=1)
        x = torch.randn(1, 8, 16, 16, requires_grad=True)
        conv(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(conv.weight.grad)


class TestDepthwiseFFTConvConstruction(unittest.TestCase):
    """Reject configurations the FFT path cannot represent."""

    def test_rejects_non_depthwise(self):
        with self.assertRaises(ValueError):
            DepthwiseFFTConv3d(8, 8, kernel_size=3, groups=1, padding=1)

    def test_rejects_bias(self):
        with self.assertRaises(ValueError):
            DepthwiseFFTConv3d(8, 8, kernel_size=3, groups=8, padding=1, bias=True)

    def test_weight_shape(self):
        conv = DepthwiseFFTConv3d(16, 16, kernel_size=3, groups=16, padding=1)
        self.assertEqual(conv.weight.shape, (16, 1, 3, 3, 3))

    def test_weight_initialised(self):
        conv = DepthwiseFFTConv3d(64, 64, kernel_size=3, groups=64, padding=1)
        # kaiming_uniform with fan_in = 1 * 3^3 = 27 → bound ≈ 1/sqrt(27) ≈ 0.19
        self.assertGreater(conv.weight.abs().max().item(), 0.0)
        self.assertLess(conv.weight.abs().max().item(), 5.0 / math.sqrt(27))


# ---------------------------------------------------------------------------
# HyenaMixer configuration validation — no CUDA required (construction only,
# but nvsubquadratic must be present to reach the validation branch)
# ---------------------------------------------------------------------------


@skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
class TestHyenaMixerConfigValidation(unittest.TestCase):
    def test_rejects_circular_with_double_grid(self):
        with self.assertRaisesRegex(ValueError, "circular.*single"):
            HyenaMixer(dim=12, spatial_dims=3, fft_padding="circular", grid_type="double")

    def test_rejects_chunked_with_circular(self):
        with self.assertRaisesRegex(ValueError, "chunked.*zero|zero.*chunked"):
            HyenaMixer(dim=12, spatial_dims=3, fft_padding="circular", use_chunked_fftconv=True)

    def test_rejects_bad_fft_padding(self):
        with self.assertRaisesRegex(ValueError, "fft_padding"):
            HyenaMixer(dim=12, spatial_dims=3, fft_padding="reflective")

    def test_rejects_bad_grid_type(self):
        with self.assertRaisesRegex(ValueError, "grid_type"):
            HyenaMixer(dim=12, spatial_dims=3, grid_type="triple")

    def test_rejects_bad_spatial_dims(self):
        with self.assertRaisesRegex(ValueError, "spatial_dims"):
            HyenaMixer(dim=12, spatial_dims=4)

    def test_zero_double_chunked_constructs(self):
        m = HyenaMixer(dim=12, spatial_dims=3, fft_padding="zero", grid_type="double", use_chunked_fftconv=True)
        self.assertEqual(m.dim, 12)


class TestHyenaMixerOptionalDep(unittest.TestCase):
    """When ``nvsubquadratic`` is missing, ``HyenaMixer`` must raise a clear ImportError."""

    @skipUnless(not HAS_NVSUBQ, "Only runs when nvsubquadratic is absent")
    def test_raises_import_error(self):
        with self.assertRaisesRegex(ImportError, "nvsubquadratic"):
            HyenaMixer(dim=12, spatial_dims=3)


# ---------------------------------------------------------------------------
# Forward shape — channels-last [B, *spatial, C] preserved
# ---------------------------------------------------------------------------


@skipUnless(HAS_NVSUBQ and HAS_CUDA, "Requires nvsubquadratic and CUDA")
class TestHyenaMixerForward(unittest.TestCase):
    device = "cuda"

    def test_3d_forward_shape(self):
        m = HyenaMixer(dim=12, spatial_dims=3).to(self.device)
        x = torch.randn(2, 8, 8, 8, 12, device=self.device)
        self.assertEqual(m(x).shape, x.shape)

    def test_2d_forward_shape(self):
        m = HyenaMixer(dim=8, spatial_dims=2).to(self.device)
        x = torch.randn(2, 16, 16, 8, device=self.device)
        self.assertEqual(m(x).shape, x.shape)

    def test_zero_padding_forward(self):
        m = HyenaMixer(dim=12, spatial_dims=3, fft_padding="zero", grid_type="single").to(self.device)
        x = torch.randn(2, 8, 8, 8, 12, device=self.device)
        self.assertEqual(m(x).shape, x.shape)

    def test_zero_double_chunked_forward(self):
        m = HyenaMixer(dim=12, spatial_dims=3, fft_padding="zero", grid_type="double", use_chunked_fftconv=True).to(
            self.device
        )
        x = torch.randn(2, 8, 8, 8, 12, device=self.device)
        self.assertEqual(m(x).shape, x.shape)


@skipUnless(HAS_NVSUBQ and HAS_CUDA, "Requires nvsubquadratic and CUDA")
class TestHyenaMixerGradients(unittest.TestCase):
    device = "cuda"

    def test_qkv_and_out_proj_get_grads(self):
        m = HyenaMixer(dim=12, spatial_dims=3).to(self.device)
        x = torch.randn(2, 6, 6, 6, 12, device=self.device, requires_grad=True)
        m(x).sum().backward()
        self.assertIsNotNone(m.qkv_proj.weight.grad)
        self.assertIsNotNone(m.out_proj.weight.grad)
        self.assertIsNotNone(x.grad)

    def test_mixer_internal_params_get_grads(self):
        m = HyenaMixer(dim=12, spatial_dims=3).to(self.device)
        x = torch.randn(2, 6, 6, 6, 12, device=self.device)
        m(x).sum().backward()
        with_grad = [
            name for name, p in m.mixer.named_parameters() if p.grad is not None and p.grad.abs().sum().item() > 0
        ]
        self.assertGreater(len(with_grad), 0, "no mixer-internal params received a gradient")


@skipUnless(HAS_NVSUBQ and HAS_CUDA, "Requires nvsubquadratic and CUDA")
class TestHyenaMixerAMP(unittest.TestCase):
    """Under ``torch.autocast`` the output dtype must match the autocast dtype."""

    device = "cuda"

    @parameterized.expand([("fp16", torch.float16), ("bf16", torch.bfloat16)])
    def test_autocast_output_dtype(self, _name, dtype):
        m = HyenaMixer(dim=12, spatial_dims=3).to(self.device)
        x = torch.randn(2, 6, 6, 6, 12, device=self.device)
        with torch.autocast("cuda", dtype=dtype):
            out = m(x)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(out.shape, x.shape)

    def test_float32_preserved(self):
        m = HyenaMixer(dim=12, spatial_dims=3).to(self.device)
        x = torch.randn(2, 6, 6, 6, 12, device=self.device)
        self.assertEqual(m(x).dtype, torch.float32)


@skipUnless(HAS_NVSUBQ and HAS_CUDA, "Requires nvsubquadratic and CUDA")
class TestHyenaMixerDeterminism(unittest.TestCase):
    device = "cuda"

    def test_same_seed_same_output(self):
        torch.manual_seed(0)
        m1 = HyenaMixer(dim=12, spatial_dims=3).to(self.device)
        torch.manual_seed(0)
        m2 = HyenaMixer(dim=12, spatial_dims=3).to(self.device)
        x = torch.randn(1, 6, 6, 6, 12, device=self.device)
        with torch.no_grad():
            y1, y2 = m1(x), m2(x)
        torch.testing.assert_close(y1, y2, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# HyenaTransformerBlock — full residual forward path
# ---------------------------------------------------------------------------


@skipUnless(HAS_NVSUBQ and HAS_CUDA, "Requires nvsubquadratic and CUDA")
class TestHyenaTransformerBlock(unittest.TestCase):
    device = "cuda"

    def test_3d_forward_shape(self):
        blk = HyenaTransformerBlock(dim=12, spatial_dims=3).to(self.device)
        x = torch.randn(2, 6, 6, 6, 12, device=self.device)
        self.assertEqual(blk(x).shape, x.shape)

    def test_2d_forward_shape(self):
        blk = HyenaTransformerBlock(dim=8, spatial_dims=2).to(self.device)
        x = torch.randn(2, 16, 16, 8, device=self.device)
        self.assertEqual(blk(x).shape, x.shape)

    def test_grad_flow_through_block(self):
        blk = HyenaTransformerBlock(dim=12, spatial_dims=3).to(self.device)
        x = torch.randn(2, 6, 6, 6, 12, device=self.device)
        blk(x).sum().backward()
        self.assertIsNotNone(blk.mixer.qkv_proj.weight.grad)
        self.assertIsNotNone(blk.mixer.out_proj.weight.grad)
        mlp_params_with_grad = [p for p in blk.mlp.parameters() if p.grad is not None]
        self.assertGreater(len(mlp_params_with_grad), 0)

    def test_mask_matrix_accepted_and_ignored(self):
        """``mask_matrix`` is accepted (signature parity with Swin) but ignored."""
        blk = HyenaTransformerBlock(dim=12, spatial_dims=3).to(self.device)
        x = torch.randn(2, 6, 6, 6, 12, device=self.device)
        with torch.no_grad():
            y1 = blk(x)
            y2 = blk(x, mask_matrix=torch.ones(1, device=self.device))
        torch.testing.assert_close(y1, y2)


@skipUnless(HAS_NVSUBQ and HAS_CUDA, "Requires nvsubquadratic and CUDA")
class TestHyenaMixerFFTShortConv(unittest.TestCase):
    """The use_fft_short_conv=True path swaps Conv3d for DepthwiseFFTConv3d."""

    device = "cuda"

    def test_3d_constructs_and_runs(self):
        m = HyenaMixer(dim=12, spatial_dims=3, use_fft_short_conv=True).to(self.device)
        x = torch.randn(2, 8, 8, 8, 12, device=self.device)
        self.assertEqual(m(x).shape, x.shape)

    def test_3d_with_short_conv_chunks(self):
        m = HyenaMixer(dim=12, spatial_dims=3, use_fft_short_conv=True, short_conv_fft_chunk_size=4).to(self.device)
        x = torch.randn(2, 8, 8, 8, 12, device=self.device)
        self.assertEqual(m(x).shape, x.shape)


if __name__ == "__main__":
    unittest.main()
