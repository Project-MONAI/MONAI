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
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import Transposed
from tests.utils import TEST_NDARRAYS

KEYS = ("i", "j")

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(
        [
            p(np.arange(5 * 4).reshape(5, 4)),
            [1, 0],
        ]
    )
    TESTS.append(
        [
            p(np.arange(5 * 4).reshape(5, 4)),
            None,
        ]
    )
    TESTS.append(
        [
            p(np.arange(5 * 4 * 3).reshape(5, 4, 3)),
            [2, 0, 1],
        ]
    )
    TESTS.append(
        [
            p(np.arange(5 * 4 * 3).reshape(5, 4, 3)),
            None,
        ]
    )


class TestTranspose(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_transpose(self, im, indices):
        im_cpu = deepcopy(im if isinstance(im, np.ndarray) else im.cpu())
        out_gt = np.transpose(im_cpu, indices)

        data = {k: deepcopy(im) for k in KEYS}
        tr = Transposed(KEYS, indices)
        out_data = tr(data)

        for k, v in out_data.items():
            if k not in KEYS:
                continue
            self.assertEqual(type(im), type(v))
            if isinstance(v, torch.Tensor):
                self.assertEqual(im.device, v.device)
                v = v.cpu()
            np.testing.assert_array_equal(v, out_gt)

        # test inverse
        fwd_inv_data = tr.inverse(out_data)

        for k, v in fwd_inv_data.items():
            if k not in KEYS:
                continue
            self.assertEqual(type(im), type(v))
            if isinstance(v, torch.Tensor):
                self.assertEqual(im.device, v.device)
                v = v.cpu()
            np.testing.assert_array_equal(v, im_cpu)


if __name__ == "__main__":
    unittest.main()
