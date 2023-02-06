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

import torch
from parameterized import parameterized

from monai.transforms.lazy.functional import resample
from monai.utils import convert_to_tensor
from tests.utils import assert_allclose, get_arange_img


def rotate_90_2d():
    t = torch.eye(3)
    t[:, 0] = torch.FloatTensor([0, -1, 0])
    t[:, 1] = torch.FloatTensor([1, 0, 0])
    return t


RESAMPLE_FUNCTION_CASES = [(get_arange_img((3, 3)), rotate_90_2d(), [[0, 3, 6], [0, 3, 6], [0, 3, 6]])]


class TestResampleFunction(unittest.TestCase):
    @parameterized.expand(RESAMPLE_FUNCTION_CASES)
    def test_resample_function_impl(self, img, matrix, expected):
        out = resample(convert_to_tensor(img), matrix, img.shape[1:])
        assert_allclose(out[0], expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
