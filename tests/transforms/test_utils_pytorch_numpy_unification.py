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

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms.utils_pytorch_numpy_unification import max, min, mode, percentile
from monai.utils import dtype_numpy_to_torch, set_determinism
from tests.test_utils import TEST_NDARRAYS, assert_allclose, skip_if_quick

TEST_MODE = []
for p in TEST_NDARRAYS:
    TEST_MODE.append([p(np.array([1, 2, 3, 4, 4, 5])), p(4), False])
    TEST_MODE.append([p(np.array([3.1, 4.1, 4.1, 5.1])), p(4.1), False])
    TEST_MODE.append([p(np.array([3.1, 4.1, 4.1, 5.1])), p(4), True])

TEST_MIN_MAX = []
for p in TEST_NDARRAYS:
    TEST_MIN_MAX.append([p(np.array([1, 2, 3, 4, 4, 5])), {}, min, p(1)])
    TEST_MIN_MAX.append([p(np.array([[3.1, 4.1, 4.1, 5.1], [3, 5, 4.1, 5]])), {"dim": 1}, min, p([3.1, 3])])
    TEST_MIN_MAX.append([p(np.array([1, 2, 3, 4, 4, 5])), {}, max, p(5)])
    TEST_MIN_MAX.append([p(np.array([[3.1, 4.1, 4.1, 5.1], [3, 5, 4.1, 5]])), {"dim": 1}, max, p([5.1, 5])])


class TestPytorchNumpyUnification(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(0)

    def test_percentile(self):
        for size in (1, 100):
            q = np.random.randint(0, 100, size=size)
            results = []
            for idx, p in enumerate(TEST_NDARRAYS):
                dtype = [np.float32, float][idx % 2]
                arr = p(np.arange(100 * 101).reshape(1, 100, 101).astype(dtype))
                results.append(percentile(arr, q))
                assert_allclose(results[0], results[-1], type_test=False, atol=1e-4, rtol=1e-4)

    @skip_if_quick
    def test_many_elements_quantile(self):  # pytorch#64947
        for elements in (1000, 17_000_000):
            data = np.random.randn(elements)
            for p in TEST_NDARRAYS:
                x = p(data)
                for t in [*TEST_NDARRAYS, list]:
                    q = percentile(x, t([10, 50]))
                    if isinstance(x, torch.Tensor):
                        self.assertIsInstance(q, torch.Tensor)
                    assert_allclose(q.shape, [2], type_test=False)

    @parameterized.expand([[p, dtype] for p in TEST_NDARRAYS for dtype in (np.uint8, np.int16, np.int32, np.int64)])
    def test_percentile_integer_input(self, p, dtype):
        # interpolated percentiles of integer inputs must not raise or be truncated
        data = np.array([[0, 1, 2, 3], [1, 0, 1, 0]]).astype(dtype)
        for kwargs in ({}, {"dim": 1}):
            result = percentile(p(data), [25, 50], **kwargs)
            expected = np.percentile(data, [25, 50], axis=kwargs.get("dim"))
            if isinstance(result, torch.Tensor):
                self.assertTrue(result.is_floating_point())
            else:
                self.assertTrue(np.issubdtype(result.dtype, np.floating))
            assert_allclose(result, expected, type_test=False, atol=1e-6)

    def test_percentile_bool_tensor(self):
        result = percentile(torch.tensor([True, False, True, False]), 50)
        self.assertTrue(result.is_floating_point())
        assert_allclose(result, 0.5, type_test=False)

    def test_percentile_integer_input_many_elements(self):
        # tensors above 1e6 elements go through `np.percentile` (pytorch#64947), the result must not be truncated
        data = np.arange(1_000_004, dtype=np.int32) % 4
        for p in TEST_NDARRAYS:
            result = percentile(p(data), 50)
            assert_allclose(result, 1.5, type_test=False)

    def test_percentile_floating_input_keeps_dtype(self):
        for p in TEST_NDARRAYS:
            for dtype in (np.float16, np.float32, np.float64):
                result = percentile(p(np.arange(10, dtype=dtype)), [25, 50])
                if isinstance(result, torch.Tensor):
                    self.assertEqual(result.dtype, dtype_numpy_to_torch(np.dtype(dtype)))
                else:
                    self.assertEqual(result.dtype, np.dtype(dtype))
                assert_allclose(result, [2.25, 4.5], type_test=False)

    def test_percentile_bool_many_elements(self):
        # booleans are not interpolated by `np.percentile`, which is used for numpy inputs and large tensors
        data = np.arange(1_000_004) % 2 == 0
        for p in TEST_NDARRAYS:
            result = percentile(p(data), 50)
            assert_allclose(result, 0.5, type_test=False)

    def test_percentile_bfloat16_many_elements(self):
        # `Tensor.numpy` does not support bfloat16, tensors above 1e6 elements are computed with numpy
        result = percentile((torch.arange(1_000_004) % 4).to(torch.bfloat16), [25, 50])
        self.assertEqual(result.dtype, torch.bfloat16)
        assert_allclose(result.float(), [0.75, 1.5], type_test=False, atol=1e-2)

    def test_percentile_bfloat16_tensor(self):
        result = percentile(torch.arange(10, dtype=torch.bfloat16), [25, 50])
        self.assertEqual(result.dtype, torch.bfloat16)
        assert_allclose(result.float(), [2.25, 4.5], type_test=False, atol=1e-2)

    def test_fails(self):
        for p in TEST_NDARRAYS:
            for q in (-1, 101):
                arr = p(np.arange(100 * 101).reshape(1, 100, 101).astype(np.float32))
                with self.assertRaises(ValueError):
                    percentile(arr, q)

    def test_dim(self):
        q = np.random.randint(0, 100, size=50)
        results = []
        for p in TEST_NDARRAYS:
            arr = p(np.arange(6).reshape(1, 2, 3).astype(np.float32))
            results.append(percentile(arr, q, dim=1))
            assert_allclose(results[0], results[-1], type_test=False, atol=1e-4)

    @parameterized.expand(TEST_MODE)
    def test_mode(self, array, expected, to_long):
        res = mode(array, to_long=to_long)
        assert_allclose(res, expected)

    @parameterized.expand(TEST_MIN_MAX)
    def test_min_max(self, array, input_params, func, expected):
        res = func(array, **input_params)
        assert_allclose(res, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
