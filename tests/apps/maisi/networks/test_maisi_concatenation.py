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
import warnings
from unittest.mock import patch

import torch
import torch.nn.functional as F
from parameterized import parameterized

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    AutoencoderKlMaisi,
    MaisiConvolution,
    MaisiGroupNorm3D,
    _concatenate_with_cpu_fallback,
)


class _BoundaryChecks:
    device = torch.device("cpu")

    @parameterized.expand([(499, False), (500, False), (512, False), (500, True)])
    def test_group_norm(self, length, norm_float16):
        image = torch.randn(1, 6, length, 2, 3, device=self.device)
        norm = MaisiGroupNorm3D(3, 6, norm_float16=norm_float16, save_mem=False).to(self.device)
        expected = F.group_norm(image, 3, norm.weight, norm.bias, norm.eps)
        if norm_float16:
            expected = expected.half()

        with patch("torch.cat", wraps=torch.cat) as concatenate:
            result = norm(image)

        torch.testing.assert_close(result, expected)
        self.assertEqual(result.device, self.device)
        self.assertEqual(concatenate.call_count, 1)
        chunks = concatenate.call_args.args[0]
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(chunk.device == self.device for chunk in chunks))

    @parameterized.expand([(499, 0), (500, 0), (512, 0), (500, 1), (500, 2)])
    def test_convolution_overlap(self, length, split_dimension):
        shape = [1, 2, 2, 2, 2]
        axis = split_dimension + 2
        shape[axis] = 6
        shape[2 + (split_dimension + 1) % 3] = length
        image = torch.randn(shape, device=self.device, requires_grad=True)
        outputs = [image.narrow(axis, start, size) for start, size in [(0, 3), (1, 4), (3, 3)]]
        convolution = MaisiConvolution(
            3, 2, 2, num_splits=3, dim_split=split_dimension, print_info=False, save_mem=False, conv_only=True
        )

        with patch("torch.cat", wraps=torch.cat) as concatenate:
            result = convolution._concatenate_tensors(outputs, split_size=2, padding=1)

        torch.testing.assert_close(result, image, rtol=0, atol=0)
        self.assertEqual(result.device, self.device)
        self.assertEqual(concatenate.call_count, 1)
        self.assertTrue(all(chunk.device == self.device for chunk in concatenate.call_args.args[0]))
        result.square().sum().backward()
        torch.testing.assert_close(image.grad, 2 * image.detach())


class TestMaisiConcatenationCPU(_BoundaryChecks, unittest.TestCase):
    def test_cpu_preserves_promoted_dtype_and_gradients(self):
        inputs = [torch.randn(2, 3, dtype=dtype, requires_grad=True) for dtype in (torch.float32, torch.float64)]
        expected = torch.cat(inputs, dim=1)

        with warnings.catch_warnings(record=True) as emitted, patch("torch.cuda.mem_get_info") as memory_info:
            warnings.simplefilter("always")
            result = _concatenate_with_cpu_fallback(inputs, 1, concat_memory_margin=100)

        memory_info.assert_not_called()
        self.assertEqual(emitted, [])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertEqual(result.dtype, torch.float64)
        gradients = torch.autograd.grad(result.square().sum(), inputs)
        for tensor, gradient in zip(inputs, gradients):
            torch.testing.assert_close(gradient, 2 * tensor.detach())

    def test_cpu_oom_is_not_retried(self):
        inputs = [torch.ones(2, 3), torch.ones(2, 3)]
        error = torch.cuda.OutOfMemoryError("simulated allocation failure")
        with patch("torch.cat", side_effect=error) as concatenate:
            with self.assertRaises(torch.cuda.OutOfMemoryError) as raised:
                _concatenate_with_cpu_fallback(inputs, 1)
        self.assertIs(raised.exception, error)
        concatenate.assert_called_once()

    @parameterized.expand([(-1,), (True,), (1.5,), ("100",)])
    def test_invalid_memory_margin(self, margin):
        with self.assertRaises(ValueError):
            MaisiGroupNorm3D(1, 2, concat_memory_margin=margin)
        with self.assertRaises(ValueError):
            MaisiConvolution(3, 2, 2, 2, 0, False, concat_memory_margin=margin)
        with self.assertRaises(ValueError):
            self._make_autoencoder(margin)

    def test_autoencoder_propagates_memory_margin(self):
        model = self._make_autoencoder(12345)
        leaves = [module for module in model.modules() if isinstance(module, (MaisiGroupNorm3D, MaisiConvolution))]
        self.assertTrue(leaves)
        for module in leaves:
            with self.subTest(module=type(module).__name__):
                self.assertEqual(module.concat_memory_margin, 12345)

    @staticmethod
    def _make_autoencoder(margin):
        return AutoencoderKlMaisi(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(2, 2),
            latent_channels=2,
            attention_levels=(False, False),
            num_res_blocks=(1, 1),
            norm_num_groups=1,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            num_splits=2,
            print_info=False,
            save_mem=False,
            concat_memory_margin=margin,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestMaisiConcatenationCUDA(_BoundaryChecks, unittest.TestCase):
    device = torch.device("cuda:0")

    @parameterized.expand([(torch.float32,), (torch.float64,)])
    def test_oom_fallback_preserves_values_device_dtype_and_gradients(self, dtype):
        inputs = [torch.randn(2, 3, device=self.device, dtype=dtype, requires_grad=True) for _ in range(3)]
        expected = torch.cat(inputs, dim=1)
        original_inputs = inputs.copy()
        concatenate = torch.cat
        attempts = []

        def simulate_cuda_oom(tensors, dim):
            attempts.append(tensors[0].device.type)
            if tensors[0].is_cuda:
                raise torch.cuda.OutOfMemoryError("simulated CUDA allocation failure")
            return concatenate(tensors, dim=dim)

        with (
            patch("torch.cat", side_effect=simulate_cuda_oom),
            self.assertWarnsRegex(
                RuntimeWarning,
                r"^CUDA concatenation ran out of memory; falling back to CPU\. This may reduce performance\.$",
            ) as warning,
        ):
            result = _concatenate_with_cpu_fallback(inputs, 1, save_mem=False)

        self.assertEqual(warning.filename, _concatenate_with_cpu_fallback.__code__.co_filename)
        self.assertEqual(attempts, ["cuda", "cpu"])
        self.assertEqual(result.device, self.device)
        self.assertEqual(result.dtype, dtype)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        gradients = torch.autograd.grad(result.square().sum(), original_inputs)
        for tensor, gradient in zip(original_inputs, gradients):
            torch.testing.assert_close(gradient, 2 * tensor.detach())

    def test_non_oom_error_is_not_retried(self):
        inputs = [torch.ones(2, 3, device=self.device), torch.ones(2, 3, device=self.device)]
        error = RuntimeError("simulated invalid concatenation")
        with patch("torch.cat", side_effect=error) as concatenate:
            with self.assertRaises(RuntimeError) as raised:
                _concatenate_with_cpu_fallback(inputs, 1, save_mem=False)
        self.assertIs(raised.exception, error)
        concatenate.assert_called_once()

    @parameterized.expand([(499, False), (500, False), (512, False), (499, True), (500, True), (512, True)])
    def test_group_norm_cpu_fallback_is_bit_exact(self, length, norm_float16):
        image = torch.randn(1, 6, length, 2, 3, device=self.device)
        norm = MaisiGroupNorm3D(3, 6, norm_float16=norm_float16, save_mem=False).to(self.device)
        expected = norm(image)
        norm.concat_memory_margin = 0

        with (
            patch("torch.cuda.mem_get_info", return_value=(0, image.numel() * image.element_size())) as memory_info,
            patch("torch.cat", wraps=torch.cat) as concatenate,
        ):
            result = norm(image)

        memory_info.assert_called_once_with(self.device)
        self.assertTrue(all(chunk.device.type == "cpu" for chunk in concatenate.call_args.args[0]))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    @parameterized.expand([(0, -1), (0, 0), (100, -1), (100, 0)])
    def test_memory_margin_uses_promoted_output_bytes(self, margin, spare_bytes):
        inputs = [torch.ones(2, 3, device=self.device, dtype=dtype) for dtype in (torch.float32, torch.float64)]
        expected = torch.cat(inputs, dim=1)
        required_bytes = expected.numel() * expected.element_size() + margin

        with (
            warnings.catch_warnings(record=True) as emitted,
            patch(
                "torch.cuda.mem_get_info", return_value=(required_bytes + spare_bytes, required_bytes * 2)
            ) as memory_info,
            patch("torch.cat", wraps=torch.cat) as concatenate,
        ):
            warnings.simplefilter("always")
            result = _concatenate_with_cpu_fallback(inputs, 1, save_mem=False, concat_memory_margin=margin)

        self.assertEqual(emitted, [])
        memory_info.assert_called_once_with(self.device)
        self.assertEqual(concatenate.call_count, 1)
        concat_device = concatenate.call_args.args[0][0].device
        self.assertEqual(concat_device.type, "cpu" if spare_bytes < 0 else "cuda")
        self.assertEqual(result.device, self.device)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_default_does_not_query_free_memory(self):
        inputs = [torch.ones(2, 3, device=self.device), torch.ones(2, 3, device=self.device)]
        with warnings.catch_warnings(record=True) as emitted, patch("torch.cuda.mem_get_info") as memory_info:
            warnings.simplefilter("always")
            result = _concatenate_with_cpu_fallback(inputs, 1, save_mem=False)
        memory_info.assert_not_called()
        self.assertEqual(emitted, [])
        self.assertEqual(result.device, self.device)

    @unittest.skipUnless(torch.cuda.device_count() > 1, "Multiple CUDA devices are required")
    def test_fallback_restores_noncurrent_device(self):
        target = torch.device("cuda:1")
        inputs = [torch.ones(2, 3, device=target), torch.ones(2, 3, device=target)]
        with torch.cuda.device(0), patch("torch.cuda.mem_get_info", return_value=(0, 100)):
            result = _concatenate_with_cpu_fallback(inputs, 1, save_mem=False, concat_memory_margin=0)
        self.assertEqual(result.device, target)


if __name__ == "__main__":
    unittest.main()
