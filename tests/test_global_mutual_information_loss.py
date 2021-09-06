# Copyright 2020 MONAI Consortium
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

from monai.losses.image_dissimilarity import GlobalMutualInformationLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASES = [
    [
        {},
        {
            "pred": torch.arange(0, 3, dtype=torch.float, device=device)[None, :, None, None, None]
            .expand(1, 3, 3, 3, 3)
            .div(3),
            "target": torch.arange(0, 3, dtype=torch.float, device=device)[None, :, None, None, None]
            .expand(1, 3, 3, 3, 3)
            .div(3),
        },
        -1.0986018,
    ],
    [
        {},
        {
            "pred": torch.arange(0, 3, dtype=torch.float, device=device)[None, :, None, None, None]
            .expand(1, 3, 3, 3, 3)
            .div(3),
            "target": torch.arange(0, 3, dtype=torch.float, device=device)[None, :, None, None, None]
            .expand(1, 3, 3, 3, 3)
            .div(3)
            ** 2,
        },
        -1.083999,
    ],
    [
        {},
        {
            "pred": torch.arange(0, 3, dtype=torch.float, device=device)[None, :, None, None].expand(1, 3, 3, 3).div(3),
            "target": torch.arange(0, 3, dtype=torch.float, device=device)[None, :, None, None]
            .expand(1, 3, 3, 3)
            .div(3)
            ** 2,
        },
        -1.083999,
    ],
    [
        {},
        {
            "pred": torch.arange(0, 3, dtype=torch.float, device=device)[None, :, None].expand(1, 3, 3).div(3),
            "target": torch.arange(0, 3, dtype=torch.float, device=device)[None, :, None].expand(1, 3, 3).div(3) ** 2,
        },
        -1.083999,
    ],
    [
        {},
        {
            "pred": torch.arange(0, 3, dtype=torch.float, device=device)[None, :].div(3),
            "target": torch.arange(0, 3, dtype=torch.float, device=device)[None, :].div(3) ** 2,
        },
        -1.083999,
    ],
    [
        {},
        {
            "pred": torch.arange(0, 3, dtype=torch.float, device=device).div(3),
            "target": torch.arange(0, 3, dtype=torch.float, device=device).div(3) ** 2,
        },
        -1.1920927e-07,
    ],
]


class TestGlobalMutualInformationLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape(self, input_param, input_data, expected_val):
        result = GlobalMutualInformationLoss(**input_param).forward(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, rtol=1e-4)

    def test_ill_shape(self):
        loss = GlobalMutualInformationLoss()
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(torch.ones((1, 2), dtype=torch.float), torch.ones((1, 3), dtype=torch.float, device=device))
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(torch.ones((1, 3, 3), dtype=torch.float), torch.ones((1, 3), dtype=torch.float, device=device))

    def test_ill_opts(self):
        pred = torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device)
        target = torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, ""):
            GlobalMutualInformationLoss(num_bins=0)(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            GlobalMutualInformationLoss(num_bins=-1)(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            GlobalMutualInformationLoss(reduction="unknown")(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            GlobalMutualInformationLoss(reduction=None)(pred, target)


if __name__ == "__main__":
    unittest.main()
