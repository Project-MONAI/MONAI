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
from parameterized import parameterized

from monai.data.utils import to_affine_nd
from tests.test_dtype_convert import TEST_NDARRAYS

TESTS = []
TESTS.append([2, np.eye(4)])


class TestToAffinend(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_to_affine_nd(self, r, affine):
        outs = []
        for p in TEST_NDARRAYS:
            for q in TEST_NDARRAYS:
                res = to_affine_nd(p(r), q(affine))
                outs.append(res.cpu() if isinstance(res, torch.Tensor) else res)
                np.testing.assert_allclose(outs[-1], outs[0])


if __name__ == "__main__":
    unittest.main()
