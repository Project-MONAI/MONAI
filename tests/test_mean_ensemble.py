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

from monai.transforms import MeanEnsemble
from tests.utils import TEST_NDARRAYS

TEST_CASE_1 = [
    {"weights": None},
    [torch.ones(2, 2, 2), torch.ones(2, 2, 2) + 2],
    torch.ones(2, 2, 2) + 1,
]

TEST_CASE_2 = [
    {"weights": None},
    torch.stack([torch.ones(2, 2, 2), torch.ones(2, 2, 2) + 2]),
    torch.ones(2, 2, 2) + 1,
]

TEST_CASE_3 = [
    {"weights": [1, 3]},
    [torch.ones(2, 2, 2), torch.ones(2, 2, 2) + 2],
    torch.ones(2, 2, 2) * 2.5,
]

TEST_CASE_4 = [
    {"weights": [[1, 3], [3, 1]]},
    [torch.ones(2, 2, 2), torch.ones(2, 2, 2) + 2],
    torch.ones(2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(2, 1, 1),
]

TEST_CASE_5 = [
    {"weights": np.array([[1, 3], [3, 1]])},
    [torch.ones(2, 2, 2), torch.ones(2, 2, 2) + 2],
    torch.ones(2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(2, 1, 1),
]

TEST_CASE_6 = [
    {"weights": torch.tensor([[[1, 3]], [[3, 1]]])},
    [torch.ones(2, 2, 2, 2), torch.ones(2, 2, 2, 2) + 2],
    torch.ones(2, 2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(1, 2, 1, 1),
]


class TestMeanEnsemble(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_value(self, input_param, img, expected_value):
        for p in TEST_NDARRAYS:
            if isinstance(img, list):
                im = [p(i) for i in img]
                im_type = type(im[0])
                im_device = im[0].device if isinstance(im[0], torch.Tensor) else None
            else:
                im = p(img)
                im_type = type(im)
                im_device = im.device if isinstance(im, torch.Tensor) else None

            result = MeanEnsemble(**input_param)(im)
            self.assertEqual(im_type, type(result))
            if isinstance(result, torch.Tensor):
                self.assertEqual(result.device, im_device)
                result = result.cpu()
            np.testing.assert_allclose(result, expected_value)

    def test_cuda_value(self):
        img = torch.stack([torch.ones(2, 2, 2, 2), torch.ones(2, 2, 2, 2) + 2])
        expected_value = torch.ones(2, 2, 2, 2) * torch.tensor([2.5, 1.5]).reshape(1, 2, 1, 1)
        if torch.cuda.is_available():
            img = img.to(torch.device("cuda:0"))
            expected_value = expected_value.to(torch.device("cuda:0"))
        result = MeanEnsemble(torch.tensor([[[1, 3]], [[3, 1]]]))(img)
        torch.testing.assert_allclose(result, expected_value)


if __name__ == "__main__":
    unittest.main()
