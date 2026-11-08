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

import numpy as np
import torch
from parameterized import parameterized

from monai.data.utils import list_data_collate
from monai.inferers import SlidingWindowInferer, SlidingWindowInfererAdapt, sliding_window_inference
from monai.inferers.utils import _compute_coords
from monai.utils import optional_import
from tests.test_utils import TEST_TORCH_AND_META_TENSORS, skip_if_no_cuda, test_is_quick

_, has_tqdm = optional_import("tqdm")

TEST_CASES = [
    [(2, 3, 16), (4,), 3, 0.25, "constant", torch.device("cpu:0")],  # 1D small roi
    [(2, 3, 16, 15, 7, 9), 4, 3, 0.25, "constant", torch.device("cpu:0")],  # 4D small roi
    [(1, 3, 16, 15, 7), (4, -1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(2, 3, 16, 15, 7), (4, -1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(3, 3, 16, 15, 7), (4, -1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(1, 3, 16, 15, 7), (20, 22, 23), 10, 0.25, "constant", torch.device("cpu:0")],  # 3D large roi
    [(2, 3, 15, 7), (2, 6), 1000, 0.25, "constant", torch.device("cpu:0")],  # 2D small roi, large batch
    [(1, 3, 16, 7), (80, 50), 7, 0.25, "constant", torch.device("cpu:0")],  # 2D large roi
    [(1, 3, 16, 15, 7), (20, 22, 23), 10, 0.5, "constant", torch.device("cpu:0")],  # 3D large overlap
    [(1, 3, 16, 7), (80, 50), 7, 0.5, "gaussian", torch.device("cpu:0")],  # 2D large overlap, gaussian
    [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "gaussian", torch.device("cpu:0")],  # 3D small roi, gaussian
    [(3, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "gaussian", torch.device("cpu:0")],  # 3D small roi, gaussian
    [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "gaussian", torch.device("cuda:0")],  # test inference on gpu if availabe
    [(1, 3, 16, 15, 7), (4, 1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
    [(5, 3, 16, 15, 7), (4, 1, 7), 3, 0.25, "constant", torch.device("cpu:0")],  # 3D small roi
]

_devices = [["cpu", "cuda:0"]] if torch.cuda.is_available() else [["cpu"]]
_windows = [
    [(2, 3, 10, 11), (7, 10), 0.8, 5],
    [(2, 3, 10, 11), (15, 12), 0, 2],
    [(2, 3, 10, 11), (10, 11), 0, 3],
    [(2, 3, 511, 237), (96, 80), 0.4, 5],
    [(2, 3, 512, 245), (96, 80), 0, 5],
    [(2, 3, 512, 245), (512, 80), 0.125, 5],
    [(2, 3, 10, 11, 12), (7, 8, 10), 0.2, 2],
]
if not test_is_quick():
    _windows += [
        [(2, 1, 125, 512, 200), (96, 97, 98), (0.4, 0.32, 0), 20],
        [(2, 1, 10, 512, 200), (96, 97, 98), (0.4, 0.12, 0), 21],
        [(2, 3, 100, 100, 200), (50, 50, 100), 0, 8],
    ]

BUFFER_CASES: list = []
for x in _windows:
    for s in (1, 3, 4):
        for d in (-1, 0, 1):
            BUFFER_CASES.extend([x, s, d, dev] for dev in itertools.product(*_devices * 3))


class TestSlidingWindowInference(unittest.TestCase):
    @parameterized.expand(BUFFER_CASES)
    def test_buffers(self, size_params, buffer_steps, buffer_dim, device_params):
        def mult_two(patch, *args, **kwargs):
            return 2.0 * patch

        img_size, roi_size, overlap, sw_batch_size = size_params
        img_device, device, sw_device = device_params
        dtype = [torch.float, torch.double][roi_size[0] % 2]  # test different input dtype
        mode = ["constant", "gaussian"][img_size[1] % 2]
        image = torch.randint(0, 255, size=img_size, dtype=dtype, device=img_device)
        sw = sliding_window_inference(
            image,
            roi_size,
            sw_batch_size,
            mult_two,
            overlap,
            mode=mode,
            sw_device=sw_device,
            device=device,
            buffer_steps=buffer_steps,
            buffer_dim=buffer_dim,
        )
        max_diff = torch.max(torch.abs(image.to(sw) - 0.5 * sw)).item()
        self.assertGreater(0.001, max_diff)

    @parameterized.expand(TEST_CASES)
    def test_sliding_window_default(self, image_shape, roi_shape, sw_batch_size, overlap, mode, device):
        n_total = np.prod(image_shape)
        if mode == "constant":
            inputs = torch.arange(n_total, dtype=torch.float).reshape(*image_shape)
        else:
            inputs = torch.ones(*image_shape, dtype=torch.float)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu:0")

        def compute(data):
            return data + 1

        if mode == "constant":
            expected_val = np.arange(n_total, dtype=np.float32).reshape(*image_shape) + 1.0
        else:
            expected_val = np.ones(image_shape, dtype=np.float32) + 1.0
        result = sliding_window_inference(inputs.to(device), roi_shape, sw_batch_size, compute, overlap, mode=mode)
        np.testing.assert_string_equal(device.type, result.device.type)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap, mode)(inputs.to(device), compute)
        np.testing.assert_string_equal(device.type, result.device.type)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    @parameterized.expand([[x] for x in TEST_TORCH_AND_META_TENSORS])
    def test_default_device(self, data_type):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = data_type(torch.ones((3, 16, 15, 7))).to(device=device)
        inputs = list_data_collate([inputs])  # make a proper batch
        roi_shape = (4, 10, 7)
        sw_batch_size = 10

        def compute(data):
            return data + 1

        inputs.requires_grad = True
        result = sliding_window_inference(inputs, roi_shape, sw_batch_size, compute)
        self.assertTrue(result.requires_grad)
        np.testing.assert_string_equal(inputs.device.type, result.device.type)
        expected_val = np.ones((1, 3, 16, 15, 7), dtype=np.float32) + 1
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val)

    @parameterized.expand(list(itertools.product(TEST_TORCH_AND_META_TENSORS, ("cpu", "cuda"), ("cpu", "cuda", None))))
    @skip_if_no_cuda
    def test_sw_device(self, data_type, device, sw_device):
        inputs = data_type(torch.ones((3, 16, 15, 7))).to(device=device)
        inputs = list_data_collate([inputs])  # make a proper batch
        roi_shape = (4, 10, 7)
        sw_batch_size = 10

        def compute(data):
            self.assertEqual(data.device.type, sw_device or device)
            return data + torch.tensor(1, device=sw_device or device)

        result = sliding_window_inference(inputs, roi_shape, sw_batch_size, compute, sw_device=sw_device, device="cpu")
        np.testing.assert_string_equal("cpu", result.device.type)
        expected_val = np.ones((1, 3, 16, 15, 7), dtype=np.float32) + 1
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    def test_sigma(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 7, 7)).to(device=device)
        roi_shape = (3, 3)
        sw_batch_size = 10

        class _Pred:
            add = 1

            def compute(self, data):
                self.add += 1
                return data + self.add

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            _Pred().compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="constant",
            sigma_scale=1.0,
        )

        expected = np.array(
            [
                [
                    [
                        [3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000],
                        [3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000],
                        [3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
                        [3.6667, 3.6667, 3.6667, 3.6667, 3.6667, 3.6667, 3.6667],
                        [4.3333, 4.3333, 4.3333, 4.3333, 4.3333, 4.3333, 4.3333],
                        [4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000],
                        [5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)
        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            _Pred().compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="gaussian",
            sigma_scale=1.0,
            progress=has_tqdm,
        )
        expected = np.array(
            [
                [
                    [
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.3271625, 3.3271623, 3.3271623, 3.3271623, 3.3271623, 3.3271623, 3.3271625],
                        [3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377],
                        [4.3271623, 4.3271623, 4.3271627, 4.3271627, 4.3271627, 4.3271623, 4.3271623],
                        [4.513757, 4.513757, 4.513757, 4.513757, 4.513757, 4.513757, 4.513757],
                        [4.9999995, 5.0, 5.0, 5.0, 5.0, 5.0, 4.9999995],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=1.0)(
            inputs, _Pred().compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=[1.0, 1.0])(
            inputs, _Pred().compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=[1.0, 1.0], cache_roi_weight_map=True
        )(inputs, _Pred().compute)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_cval(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 3, 3)).to(device=device)
        roi_shape = (5, 5)
        sw_batch_size = 10

        def compute(data):
            return data + data.sum()

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="constant",
            sigma_scale=1.0,
        )
        expected = np.ones((1, 1, 3, 3)) * -6.0
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1)(inputs, compute)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_args_kwargs(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 3, 3)).to(device=device)
        t1 = torch.ones(1).to(device=device)
        t2 = torch.ones(1).to(device=device)
        roi_shape = (5, 5)
        sw_batch_size = 10

        def compute(data, test1, test2):
            return data + test1 + test2

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            0.5,
            "constant",
            1.0,
            "constant",
            0.0,
            device,
            device,
            has_tqdm,
            None,
            None,
            None,
            0,
            False,
            t1,
            test2=t2,
        )
        expected = np.ones((1, 1, 3, 3)) + 2.0
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute, t1, test2=t2)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInfererAdapt(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute, t1, test2=t2)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_multioutput(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 6, 20, 20)).to(device=device)
        roi_shape = (8, 8)
        sw_batch_size = 10

        def compute(data):
            return data + 1, data[:, ::3, ::2, ::2] + 2, data[:, ::2, ::4, ::4] + 3

        def compute_dict(data):
            return {1: data + 1, 2: data[:, ::3, ::2, ::2] + 2, 3: data[:, ::2, ::4, ::4] + 3}

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            0.5,
            "constant",
            1.0,
            "constant",
            0.0,
            device,
            device,
            has_tqdm,
            None,
        )
        result_dict = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute_dict,
            0.5,
            "constant",
            1.0,
            "constant",
            0.0,
            device,
            device,
            has_tqdm,
            None,
        )
        expected = (np.ones((1, 6, 20, 20)) + 1, np.ones((1, 2, 10, 10)) + 2, np.ones((1, 3, 5, 5)) + 3)
        expected_dict = {1: np.ones((1, 6, 20, 20)) + 1, 2: np.ones((1, 2, 10, 10)) + 2, 3: np.ones((1, 3, 5, 5)) + 3}
        for rr, ee in zip(result, expected):
            np.testing.assert_allclose(rr.cpu().numpy(), ee, rtol=1e-4)
        for rr, _ in zip(result_dict, expected_dict):
            np.testing.assert_allclose(result_dict[rr].cpu().numpy(), expected_dict[rr], rtol=1e-4)

        result = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute)
        for rr, ee in zip(result, expected):
            np.testing.assert_allclose(rr.cpu().numpy(), ee, rtol=1e-4)

        result_dict = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute_dict)
        for rr, _ in zip(result_dict, expected_dict):
            np.testing.assert_allclose(result_dict[rr].cpu().numpy(), expected_dict[rr], rtol=1e-4)

    def test_strict_shape_validation(self):
        """Test strict shape validation to ensure inputs match roi_size dimensions."""
        device = "cpu"
        roi_size = (16, 16, 16)
        sw_batch_size = 4

        def predictor(data):
            return data

        # Case 1: Input has fewer dimensions than expected (e.g., missing Batch or Channel)
        # 3D roi_size requires 5D input (B, C, D, H, W), giving 4D here.
        inputs_4d = torch.randn((1, 16, 16, 16), device=device)
        with self.assertRaisesRegex(ValueError, "Inputs must have 5 dimensions"):
            sliding_window_inference(inputs_4d, roi_size, sw_batch_size, predictor)

        # Case 2: Input is 3D (missing Batch AND Channel)
        inputs_3d = torch.randn((16, 16, 16), device=device)
        with self.assertRaisesRegex(ValueError, "Inputs must have 5 dimensions"):
            sliding_window_inference(inputs_3d, roi_size, sw_batch_size, predictor)


class TestSlidingWindowInferenceCond(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_sliding_window_default(self, image_shape, roi_shape, sw_batch_size, overlap, mode, device):
        n_total = np.prod(image_shape)
        if mode == "constant":
            inputs = torch.arange(n_total, dtype=torch.float).reshape(*image_shape)
        else:
            inputs = torch.ones(*image_shape, dtype=torch.float)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu:0")

        # condition
        condition = torch.ones(*image_shape, dtype=torch.float)

        def compute(data, condition):
            return data + condition

        if mode == "constant":
            expected_val = np.arange(n_total, dtype=np.float32).reshape(*image_shape) + 1.0
        else:
            expected_val = np.ones(image_shape, dtype=np.float32) + 1.0

        result = sliding_window_inference(
            inputs.to(device), roi_shape, sw_batch_size, compute, overlap, mode=mode, condition=condition.to(device)
        )
        np.testing.assert_string_equal(device.type, result.device.type)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap, mode)(
            inputs.to(device), compute, condition=condition.to(device)
        )
        np.testing.assert_string_equal(device.type, result.device.type)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    @parameterized.expand([[x] for x in TEST_TORCH_AND_META_TENSORS])
    def test_default_device(self, data_type):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = data_type(torch.ones((3, 16, 15, 7))).to(device=device)
        condition = torch.ones((3, 16, 15, 7)).to(device=device)
        inputs = list_data_collate([inputs])  # make a proper batch
        condition = list_data_collate([condition])  # make a proper batch
        roi_shape = (4, 10, 7)
        sw_batch_size = 10

        def compute(data, condition):
            return data + condition

        inputs.requires_grad = True
        result = sliding_window_inference(inputs, roi_shape, sw_batch_size, compute, condition=condition)
        self.assertTrue(result.requires_grad)
        np.testing.assert_string_equal(inputs.device.type, result.device.type)
        expected_val = np.ones((1, 3, 16, 15, 7), dtype=np.float32) + 1
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val)

    @parameterized.expand(list(itertools.product(TEST_TORCH_AND_META_TENSORS, ("cpu", "cuda"), ("cpu", "cuda", None))))
    @skip_if_no_cuda
    def test_sw_device(self, data_type, device, sw_device):
        inputs = data_type(torch.ones((3, 16, 15, 7))).to(device=device)
        condition = torch.ones((3, 16, 15, 7)).to(device=device)
        inputs = list_data_collate([inputs])  # make a proper batch
        condition = list_data_collate([condition])  # make a proper batch
        roi_shape = (4, 10, 7)
        sw_batch_size = 10

        def compute(data, condition):
            self.assertEqual(data.device.type, sw_device or device)
            self.assertEqual(condition.device.type, sw_device or device)

            return data + condition

        result = sliding_window_inference(
            inputs, roi_shape, sw_batch_size, compute, sw_device=sw_device, device="cpu", condition=condition
        )
        np.testing.assert_string_equal("cpu", result.device.type)
        expected_val = np.ones((1, 3, 16, 15, 7), dtype=np.float32) + 1
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    def test_sigma(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 7, 7)).to(device=device)
        roi_shape = (3, 3)
        sw_batch_size = 10

        class _Pred:
            add = 1

            def compute(self, data):
                self.add += 1
                return data + self.add

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            _Pred().compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="constant",
            sigma_scale=1.0,
        )

        expected = np.array(
            [
                [
                    [
                        [3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000],
                        [3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000],
                        [3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
                        [3.6667, 3.6667, 3.6667, 3.6667, 3.6667, 3.6667, 3.6667],
                        [4.3333, 4.3333, 4.3333, 4.3333, 4.3333, 4.3333, 4.3333],
                        [4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000],
                        [5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)
        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            _Pred().compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="gaussian",
            sigma_scale=1.0,
            progress=has_tqdm,
        )
        expected = np.array(
            [
                [
                    [
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.3271625, 3.3271623, 3.3271623, 3.3271623, 3.3271623, 3.3271623, 3.3271625],
                        [3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377],
                        [4.3271623, 4.3271623, 4.3271627, 4.3271627, 4.3271627, 4.3271623, 4.3271623],
                        [4.513757, 4.513757, 4.513757, 4.513757, 4.513757, 4.513757, 4.513757],
                        [4.9999995, 5.0, 5.0, 5.0, 5.0, 5.0, 4.9999995],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=1.0)(
            inputs, _Pred().compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=[1.0, 1.0])(
            inputs, _Pred().compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=[1.0, 1.0], cache_roi_weight_map=True
        )(inputs, _Pred().compute)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_cval(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 3, 3)).to(device=device)
        condition = torch.ones((1, 1, 3, 3)).to(device=device)
        roi_shape = (5, 5)
        sw_batch_size = 10

        def compute(data, condition):
            return data + data.sum() + condition

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="constant",
            sigma_scale=1.0,
            condition=condition,
        )
        expected = np.ones((1, 1, 3, 3)) * -6.0 + 1.0
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1)(
            inputs, compute, condition=condition
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_args_kwargs(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 3, 3)).to(device=device)
        condition = torch.ones((1, 1, 3, 3)).to(device=device)
        t1 = torch.ones(1).to(device=device)
        t2 = torch.ones(1).to(device=device)
        roi_shape = (5, 5)
        sw_batch_size = 10

        def compute(data, test1, test2, condition):
            return data + test1 + test2 + condition

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            0.5,
            "constant",
            1.0,
            "constant",
            0.0,
            device,
            device,
            has_tqdm,
            None,
            None,
            None,
            0,
            False,
            t1,
            condition=condition,
            test2=t2,
        )
        expected = np.ones((1, 1, 3, 3)) + 3.0
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute, t1, condition=condition, test2=t2)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInfererAdapt(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute, t1, condition=condition, test2=t2)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_multioutput(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 6, 20, 20)).to(device=device)
        condition = torch.ones((1, 6, 20, 20)).to(device=device)
        roi_shape = (8, 8)
        sw_batch_size = 10

        def compute(data, condition):
            return (
                data + 1 + condition,
                data[:, ::3, ::2, ::2] + 2 + condition[:, ::3, ::2, ::2],
                data[:, ::2, ::4, ::4] + 3 + condition[:, ::2, ::4, ::4],
            )

        def compute_dict(data, condition):
            return {
                1: data + 1 + condition,
                2: data[:, ::3, ::2, ::2] + 2 + condition[:, ::3, ::2, ::2],
                3: data[:, ::2, ::4, ::4] + 3 + condition[:, ::2, ::4, ::4],
            }

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            0.5,
            "constant",
            1.0,
            "constant",
            0.0,
            device,
            device,
            has_tqdm,
            None,
            condition=condition,
        )
        result_dict = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute_dict,
            0.5,
            "constant",
            1.0,
            "constant",
            0.0,
            device,
            device,
            has_tqdm,
            None,
            condition=condition,
        )
        expected = (np.ones((1, 6, 20, 20)) + 2, np.ones((1, 2, 10, 10)) + 3, np.ones((1, 3, 5, 5)) + 4)
        expected_dict = {1: np.ones((1, 6, 20, 20)) + 2, 2: np.ones((1, 2, 10, 10)) + 3, 3: np.ones((1, 3, 5, 5)) + 4}
        for rr, ee in zip(result, expected):
            np.testing.assert_allclose(rr.cpu().numpy(), ee, rtol=1e-4)
        for rr, _ in zip(result_dict, expected_dict):
            np.testing.assert_allclose(result_dict[rr].cpu().numpy(), expected_dict[rr], rtol=1e-4)

        result = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute, condition=condition)
        for rr, ee in zip(result, expected):
            np.testing.assert_allclose(rr.cpu().numpy(), ee, rtol=1e-4)

        result_dict = SlidingWindowInferer(
            roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1, progress=has_tqdm
        )(inputs, compute_dict, condition=condition)
        for rr, _ in zip(result_dict, expected_dict):
            np.testing.assert_allclose(result_dict[rr].cpu().numpy(), expected_dict[rr], rtol=1e-4)

    @parameterized.expand([(1,), (4,)])
    def test_conditioned_branches_and_buffered_parity(self, sw_batch_size):
        """Validate conditioned parity between buffered and non-buffered flows.

        Args:
            sw_batch_size (int): Sliding-window batch size.

        Returns:
            None.

        Raises:
            AssertionError: If device, conditioning alignment, or output parity checks fail.
        """
        inputs = torch.arange(1 * 1 * 10 * 8, dtype=torch.float).reshape(1, 1, 10, 8)
        condition = inputs + 100.0
        roi_shape = (4, 4)

        def compute(data, condition):
            """Compute output for a conditioned patch.

            Args:
                data (torch.Tensor): Input patch tensor.
                condition (torch.Tensor): Conditioning patch tensor aligned to ``data``.

            Returns:
                torch.Tensor: Element-wise ``data + condition``.

            Raises:
                AssertionError: If device placement or conditioning alignment checks fail.
            """
            self.assertEqual(data.device.type, "cpu")
            self.assertEqual(condition.device.type, "cpu")
            torch.testing.assert_close(condition - data, torch.full_like(data, 100.0))
            return data + condition

        # Non-buffered flow.
        result_non_buffered = sliding_window_inference(
            inputs, roi_shape, sw_batch_size, compute, overlap=0.5, mode="constant", condition=condition
        )
        # Buffered flow; should match the non-buffered output.
        result_buffered = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            overlap=0.5,
            mode="constant",
            condition=condition,
            buffer_steps=2,
            buffer_dim=0,
        )

        expected = inputs + condition
        torch.testing.assert_close(result_non_buffered, expected)
        torch.testing.assert_close(result_buffered, expected)
        torch.testing.assert_close(result_buffered, result_non_buffered)


class TestSlidingWindowUtils(unittest.TestCase):
    """Tests for low-level sliding-window utility helpers.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    def test_compute_coords_accepts_list_indices(self):
        """Ensure ``_compute_coords`` handles list-based index containers.

        Args:
            None.

        Returns:
            None.

        Raises:
            AssertionError: If computed output placement differs from expected placement.
        """
        out = torch.zeros((1, 1, 12, 12), dtype=torch.float)
        patch = torch.arange(16, dtype=torch.float).reshape(1, 1, 4, 4)
        coords = [[slice(0, 1), slice(None), slice(1, 3), slice(2, 4)]]

        _compute_coords(coords=coords, z_scale=[2.0, 2.0], out=out, patch=patch)

        expected = torch.zeros_like(out)
        expected[0, 0, 2:6, 4:8] = patch[0, 0]
        torch.testing.assert_close(out, expected)


class TestSlidingWindowStreamingSlab(unittest.TestCase):
    """Tests for Streaming Active Slab Sliding Window Inference."""

    def test_streaming_slab_constant_gaussian(self):
        """Verify identical numerical output between streaming slab and monolithic baseline."""
        for mode in ("constant", "gaussian"):
            for shape in ((1, 2, 16, 16, 16), (2, 1, 15, 17, 19)):
                x = torch.randn(*shape, dtype=torch.float32)
                roi = (8, 8, 8)

                def compute(p):
                    return p * 1.5 + 0.25

                out_mono = sliding_window_inference(x, roi, 2, compute, mode=mode, overlap=0.25)
                out_slab = sliding_window_inference(x, roi, 2, compute, mode=mode, overlap=0.25, streaming_slab=True)
                diff = torch.max(torch.abs(out_mono - out_slab)).item()
                self.assertLess(diff, 1e-4)

    def test_streaming_slab_spatial_dims(self):
        """Verify streaming slab along arbitrary slab dimensions (0, 1, 2)."""
        x = torch.randn(1, 2, 16, 16, 16, dtype=torch.float32)
        roi = (8, 8, 8)

        def compute(p):
            return p * 2.0

        out_mono = sliding_window_inference(x, roi, 2, compute, mode="gaussian", overlap=0.25)
        for sdim in (0, 1, 2, -1):
            out_slab = sliding_window_inference(
                x, roi, 2, compute, mode="gaussian", overlap=0.25, streaming_slab=True, slab_dim=sdim
            )
            diff = torch.max(torch.abs(out_mono - out_slab)).item()
            self.assertLess(diff, 1e-4)

    def test_streaming_slab_multi_output_dict_and_tuple(self):
        """Verify multi-output dictionary and tuple predictors with streaming slab."""
        x = torch.randn(1, 2, 16, 16, 16, dtype=torch.float32)
        roi = (8, 8, 8)

        # Dict output
        def dict_net(p):
            return {"seg": p * 2.0, "prob": p.sigmoid()}

        out_mono_d = sliding_window_inference(x, roi, 2, dict_net, mode="gaussian")
        out_slab_d = sliding_window_inference(x, roi, 2, dict_net, mode="gaussian", streaming_slab=True)
        self.assertLess(torch.max(torch.abs(out_mono_d["seg"] - out_slab_d["seg"])).item(), 1e-4)
        self.assertLess(torch.max(torch.abs(out_mono_d["prob"] - out_slab_d["prob"])).item(), 1e-4)

        # Tuple output
        def tuple_net(p):
            return (p * 2.0, p.sigmoid())

        out_mono_t = sliding_window_inference(x, roi, 2, tuple_net, mode="gaussian")
        out_slab_t = sliding_window_inference(x, roi, 2, tuple_net, mode="gaussian", streaming_slab=True)
        self.assertLess(torch.max(torch.abs(out_mono_t[0] - out_slab_t[0])).item(), 1e-4)
        self.assertLess(torch.max(torch.abs(out_mono_t[1] - out_slab_t[1])).item(), 1e-4)

    def test_streaming_slab_inferer_class(self):
        """Verify SlidingWindowInferer class forwards streaming_slab and slab_dim."""
        x = torch.randn(1, 2, 16, 16, 16, dtype=torch.float32)
        roi = (8, 8, 8)

        def compute(p):
            return p * 2.0

        out_mono = sliding_window_inference(x, roi, 2, compute, mode="gaussian", overlap=0.25)
        inferer = SlidingWindowInferer(roi, sw_batch_size=2, overlap=0.25, mode="gaussian", streaming_slab=True)
        out_inferer = inferer(x, compute)
        diff = torch.max(torch.abs(out_mono - out_inferer)).item()
        self.assertLess(diff, 1e-4)

    def test_streaming_slab_validation_errors(self):
        """Verify fail-early parameter checks for streaming slab."""
        x = torch.randn(1, 1, 16, 16, 16)

        def compute(p):
            return p

        # Invalid slab_dim
        with self.assertRaises(ValueError):
            sliding_window_inference(x, (8, 8, 8), 1, compute, streaming_slab=True, slab_dim=5)

        # Conflicting buffer_steps and streaming_slab
        with self.assertRaises(ValueError):
            sliding_window_inference(x, (8, 8, 8), 1, compute, buffer_steps=2, streaming_slab=True)

    def test_streaming_slab_empty_outputs(self):
        """Verify empty predictor outputs (tuple or dict) do not raise IndexError."""
        x = torch.randn(1, 1, 16, 16, 16)
        roi = (8, 8, 8)

        def empty_tuple(p):
            return ()

        def empty_dict(p):
            return {}

        out_tuple = sliding_window_inference(x, roi, 2, empty_tuple, streaming_slab=True)
        self.assertEqual(out_tuple, ())

        out_dict = sliding_window_inference(x, roi, 2, empty_dict, streaming_slab=True)
        self.assertEqual(out_dict, {})

    def test_streaming_slab_adapt_class(self):
        """Verify SlidingWindowInfererAdapt works seamlessly with streaming_slab."""
        from monai.inferers import SlidingWindowInfererAdapt

        x = torch.randn(1, 1, 16, 16, 16)
        roi = (8, 8, 8)

        def compute(p):
            return p * 2.0

        inferer = SlidingWindowInfererAdapt(roi, sw_batch_size=2, streaming_slab=True)
        out = inferer(x, compute)
        self.assertEqual(out.shape, x.shape)

        # Test call-time device override
        out_dev = inferer(x, compute, device="cpu")
        self.assertEqual(out_dev.shape, x.shape)

        # Test buffer_steps + streaming_slab raises ValueError on both adaptive (device=None) and explicit device paths
        adapt_err = SlidingWindowInfererAdapt(roi, sw_batch_size=2, streaming_slab=True, buffer_steps=2)
        with self.assertRaises(ValueError):
            adapt_err(x, compute)
        adapt_dev_err = SlidingWindowInfererAdapt(
            roi, sw_batch_size=2, streaming_slab=True, buffer_steps=2, device="cpu"
        )
        with self.assertRaises(ValueError):
            adapt_dev_err(x, compute)

    def test_streaming_slab_process_fn_spatial_shape(self):
        """Verify process_fn receives spatial-only shape (roi_size) in streaming mode, matching ordinary inference."""
        x = torch.randn(1, 1, 16, 16, 16)
        roi = (8, 8, 8)
        shapes_received = []

        def my_process_fn(seg_tuple, win_data, imp_map):
            shapes_received.append(imp_map.shape)
            return seg_tuple, imp_map

        def compute(p):
            return p

        out_mono = sliding_window_inference(x, roi, 2, compute, process_fn=my_process_fn, mode="gaussian")
        shapes_mono = list(shapes_received)
        shapes_received.clear()

        out_slab = sliding_window_inference(
            x, roi, 2, compute, process_fn=my_process_fn, mode="gaussian", streaming_slab=True
        )
        shapes_slab = list(shapes_received)

        # In both modes, every importance map passed to process_fn must have spatial-only shape (8, 8, 8), not 5D
        self.assertTrue(all(s == torch.Size(roi) for s in shapes_mono))
        self.assertTrue(all(s == torch.Size(roi) for s in shapes_slab))
        self.assertLess(torch.max(torch.abs(out_mono - out_slab)).item(), 1e-4)

    def test_streaming_slab_positional_and_kwargs(self):
        """Verify predictor extra positional args and kwargs are forwarded correctly."""
        x = torch.randn(1, 1, 16, 16, 16)
        roi = (8, 8, 8)

        def my_predictor(p, scale_factor, bias=0.0):
            return p * scale_factor + bias

        inferer_mono = SlidingWindowInferer(roi, sw_batch_size=2)
        out_mono = inferer_mono(x, my_predictor, 3.0, bias=5.0)

        inferer_slab = SlidingWindowInferer(roi, sw_batch_size=2, streaming_slab=True)
        out_slab = inferer_slab(x, my_predictor, 3.0, bias=5.0)

        self.assertLess(torch.max(torch.abs(out_mono - out_slab)).item(), 1e-4)

    def test_streaming_slab_mixed_resolution_heads(self):
        """Verify multi-head outputs with differing spatial resolutions accumulate correctly."""
        x = torch.randn(1, 1, 16, 16, 16)
        roi = (8, 8, 8)

        def multi_res_net(p):
            h0 = p * 2.0
            h1 = torch.nn.functional.interpolate(p, size=(4, 4, 4), mode="nearest")
            return {"full": h0, "half": h1}

        out_mono = sliding_window_inference(x, roi, 2, multi_res_net, mode="gaussian")
        out_slab = sliding_window_inference(x, roi, 2, multi_res_net, mode="gaussian", streaming_slab=True)

        self.assertEqual(out_slab["full"].shape, (1, 1, 16, 16, 16))
        self.assertEqual(out_slab["half"].shape, (1, 1, 8, 8, 8))
        self.assertLess(torch.max(torch.abs(out_mono["full"] - out_slab["full"])).item(), 1e-4)
        self.assertLess(torch.max(torch.abs(out_mono["half"] - out_slab["half"])).item(), 1e-4)

    def test_streaming_slab_uncovered_nan(self):
        """Verify uncovered voxels produce NaN consistently with ordinary inference."""
        x = torch.randn(1, 1, 16, 16, 16)
        roi = (8, 8, 8)

        imp = torch.zeros(roi)
        imp[2:6, 2:6, 2:6] = 1.0

        def compute(p):
            return p

        out_mono = sliding_window_inference(x, roi, 2, compute, roi_weight_map=imp, mode="constant")
        out_slab = sliding_window_inference(
            x, roi, 2, compute, roi_weight_map=imp, mode="constant", streaming_slab=True
        )

        nan_mono = torch.isnan(out_mono)
        nan_slab = torch.isnan(out_slab)
        self.assertTrue(torch.equal(nan_mono, nan_slab))
        self.assertLess(torch.max(torch.abs(out_mono[~nan_mono] - out_slab[~nan_slab])).item(), 1e-4)

    def test_streaming_slab_backward(self):
        """Verify autograd backward works through streaming slab without in-place modification errors."""
        x = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
        roi = (8, 8, 8)

        def compute(p):
            return p * 2.0

        out = sliding_window_inference(x, roi, 2, compute, streaming_slab=True)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_streaming_slab_batch_indexed_weights(self):
        """Verify process_fn returning per-window weight maps in multi-window chunks."""
        x = torch.randn(1, 1, 16, 16, 16)
        roi = (8, 8, 8)

        def my_process_fn(seg_tuple, win_data, imp_map):
            b_size = win_data.shape[0]
            weights = imp_map.expand(b_size, 1, *roi).clone()
            return seg_tuple, weights

        def compute(p):
            return p * 2.0

        out_slab = sliding_window_inference(x, roi, 2, compute, process_fn=my_process_fn, streaming_slab=True)
        self.assertEqual(out_slab.shape, x.shape)
        self.assertFalse(torch.isnan(out_slab).any())

    def test_streaming_slab_batch_sensitive_predictor(self):
        """Verify window grouping differences for batch-sensitive predictors."""
        x = torch.ones(1, 1, 8)
        roi = (4,)

        def compute(p):
            return p + p.sum()

        out_mono = sliding_window_inference(x, roi, 2, compute, overlap=0.0)
        out_slab = sliding_window_inference(x, roi, 2, compute, overlap=0.0, streaming_slab=True)
        self.assertEqual(out_slab[0, 0, 0].item(), 5.0)
        self.assertEqual(out_mono[0, 0, 0].item(), 9.0)


if __name__ == "__main__":
    unittest.main()
