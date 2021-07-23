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

from monai.transforms import VoteEnsembled
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    # shape: [1, 2, 1, 1]
    TESTS.append(
        [
            p,
            {"keys": ["pred0", "pred1", "pred2"], "output_key": "output", "num_classes": None},
            {
                "pred0": torch.tensor([[[[1]], [[0]]]]),
                "pred1": torch.tensor([[[[1]], [[0]]]]),
                "pred2": torch.tensor([[[[0]], [[1]]]]),
            },
            torch.tensor([[[[1.0]], [[0.0]]]]),
        ]
    )

    # shape: [1, 2, 1, 1]
    TESTS.append(
        [
            p,
            {"keys": "output", "output_key": "output", "num_classes": None},
            {
                "output": torch.stack(
                    [torch.tensor([[[[1]], [[0]]]]), torch.tensor([[[[1]], [[0]]]]), torch.tensor([[[[0]], [[1]]]])]
                )
            },
            torch.tensor([[[[1.0]], [[0.0]]]]),
        ]
    )

    # shape: [1, 2, 1]
    TESTS.append(
        [
            p,
            {"keys": ["pred0", "pred1", "pred2"], "output_key": "output", "num_classes": 3},
            {
                "pred0": torch.tensor([[[0], [2]]]),
                "pred1": torch.tensor([[[0], [2]]]),
                "pred2": torch.tensor([[[1], [1]]]),
            },
            torch.tensor([[[0], [2]]]),
        ]
    )

    # shape: [1, 2, 1]
    TESTS.append(
        [
            p,
            {"keys": ["pred0", "pred1", "pred2"], "output_key": "output", "num_classes": 5},
            {
                "pred0": torch.tensor([[[0], [2]]]),
                "pred1": torch.tensor([[[0], [2]]]),
                "pred2": torch.tensor([[[1], [1]]]),
            },
            torch.tensor([[[0], [2]]]),
        ]
    )

    # shape: [1]
    TESTS.append(
        [
            p,
            {"keys": ["pred0", "pred1", "pred2"], "output_key": "output", "num_classes": 3},
            {"pred0": torch.tensor([2]), "pred1": torch.tensor([2]), "pred2": torch.tensor([1])},
            torch.tensor([2]),
        ]
    )


class TestVoteEnsembled(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, in_type, input_param, img, expected_value):
        for k, v in img.items():
            img[k] = in_type(v)
        result = VoteEnsembled(**input_param)(img)["output"]
        in_im = img["pred0"] if "pred0" in img else img["output"]
        self.assertEqual(type(result), type(in_im))
        if isinstance(result, torch.Tensor):
            self.assertEqual(result.device, in_im.device)
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
        result = VoteEnsembled(keys="output", num_classes=None)({"output": img})
        torch.testing.assert_allclose(result["output"], expected_value)


if __name__ == "__main__":
    unittest.main()
