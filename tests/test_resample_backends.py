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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data import MetaTensor
from monai.transforms import Resample
from monai.transforms.utils import create_grid
from monai.utils import convert_to_numpy
from tests.utils import assert_allclose, is_tf32_env

_rtol = 1e-3 if is_tf32_env() else 1e-4

TEST_IDENTITY = []
for p in (int, float, np.float16, np.float32, np.float64):
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TEST_IDENTITY.append([dict(device=device), p, (1, 3, 5)])
        TEST_IDENTITY.append([dict(device=device), p, (1, 3, 5, 8)])


class TestResampleBackends(unittest.TestCase):
    @parameterized.expand(TEST_IDENTITY)
    def test_resample_identity(self, input_param, im_type, input_shape):
        xform = Resample(**input_param)
        n_elem = np.prod(input_shape)
        img = convert_to_numpy(np.arange(n_elem).reshape(input_shape), dtype=im_type)
        grid = create_grid(input_shape[1:], homogeneous=True)
        output = xform(img=img, grid=grid)
        self.assertIsInstance(output, MetaTensor)
        print(output.dtype)
        assert_allclose(output, img, rtol=_rtol, atol=1e-5, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
