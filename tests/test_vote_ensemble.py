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

from monai.transforms import VoteEnsemble
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    # shape: [2, 1, 1]
    TESTS.append(
        [
            p,
            {"num_classes": None},
            [torch.tensor([[[1]], [[0]]]), torch.tensor([[[1]], [[0]]]), torch.tensor([[[0]], [[1]]])],
            torch.tensor([[[1.0]], [[0.0]]]),
        ]
    )
    TESTS.append(
        # shape: [1, 2, 1, 1]
        [
            p,
            {"num_classes": None},
            torch.stack(
                [torch.tensor([[[[1]], [[0]]]]), torch.tensor([[[[1]], [[0]]]]), torch.tensor([[[[0]], [[1]]]])]
            ),
            torch.tensor([[[[1.0]], [[0.0]]]]),
        ]
    )
    TESTS.append(
        # shape: [1, 2, 1]
        [
            p,
            {"num_classes": 3},
            [torch.tensor([[[0], [2]]]), torch.tensor([[[0], [2]]]), torch.tensor([[[1], [1]]])],
            torch.tensor([[[0], [2]]]),
        ]
    )
    TESTS.append(
        # shape: [1, 2, 1]
        [
            p,
            {"num_classes": 5},
            [torch.tensor([[[0], [2]]]), torch.tensor([[[0], [2]]]), torch.tensor([[[1], [1]]])],
            torch.tensor([[[0], [2]]]),
        ]
    )
    TESTS.append(
        # shape: [1]
        [
            p,
            {"num_classes": 3},
            [torch.tensor([2]), torch.tensor([2]), torch.tensor([1])],
            torch.tensor([2]),
        ]
    )
    TESTS.append(
        # shape: 1
        [
            p,
            {"num_classes": 3},
            [torch.tensor(2), torch.tensor(2), torch.tensor(1)],
            torch.tensor(2),
        ]
    )


class TestVoteEnsemble(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, in_type, input_param, img, expected_value):
        if isinstance(img, list):
            im = [in_type(i) for i in img]
            im_type = type(im[0])
            im_device = im[0].device if isinstance(im[0], torch.Tensor) else None
        else:
            im = in_type(img)
            im_type = type(im)
            im_device = im.device if isinstance(im, torch.Tensor) else None

        result = VoteEnsemble(**input_param)(im)
        self.assertEqual(im_type, type(result))
        if isinstance(result, torch.Tensor):
            self.assertEqual(result.device, im_device)
            result = result.cpu()
        np.testing.assert_allclose(result, expected_value)

    def test_cuda_value(self):
        img = torch.stack(
            [torch.tensor([[[[1]], [[0]]]]), torch.tensor([[[[1]], [[0]]]]), torch.tensor([[[[0]], [[1]]]])]
        )
        expected_value = torch.tensor([[[[1.0]], [[0.0]]]])
        if torch.cuda.is_available():
            img = img.to(torch.device("cuda:0"))
            expected_value = expected_value.to(torch.device("cuda:0"))
        result = VoteEnsemble(num_classes=None)(img)
        torch.testing.assert_allclose(result, expected_value)


if __name__ == "__main__":
    unittest.main()
