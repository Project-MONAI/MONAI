# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import torch

from monai.transforms.utils_pytorch_numpy_unification import percentile
from tests.utils import TEST_NDARRAYS, assert_allclose, set_determinism


class TestPytorchNumpyUnification(unittest.TestCase):
    def setUp(self) -> None:
        set_determinism(0)

    def test_percentile(self):
        for size in (1, 100):
            q = np.random.randint(0, 100, size=size)
            results = []
            for p in TEST_NDARRAYS:
                arr = p(np.arange(100 * 101).reshape(1, 100, 101).astype(np.float32))
                results.append(percentile(arr, q))
                # pre torch 1.7, no `quantile`. Our own method doesn't interpolate,
                # so we can only be accurate to 0.5
                atol = 0.5 if not hasattr(torch, "quantile") else 1e-4
                assert_allclose(results[0], results[-1], type_test=False, atol=atol)

    def test_fails(self):
        for p in TEST_NDARRAYS:
            for q in (-1, 101):
                arr = p(np.arange(100 * 101).reshape(1, 100, 101).astype(np.float32))
                with self.assertRaises(ValueError):
                    percentile(arr, q)


if __name__ == "__main__":
    unittest.main()
