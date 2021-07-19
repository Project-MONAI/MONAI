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

from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric, compute_roc_auc
from monai.transforms import Activations, AsDiscrete, Compose, ToTensor
from tests.utils import TEST_NDARRAYS

TESTS = []
for p in TEST_NDARRAYS:
    for q in TEST_NDARRAYS:
        TESTS.append(
            [
                p(torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5]])),
                q(torch.tensor([[0], [1], [0], [1]])),
                True,
                True,
                "macro",
                0.75,
            ]
        )
        TESTS.append(
            [
                p(torch.tensor([[0.5], [0.5], [0.2], [8.3]])),
                q(torch.tensor([[0], [1], [0], [1]])),
                False,
                False,
                "macro",
                0.875,
            ]
        )
        TESTS.append(
            [
                p(torch.tensor([[0.5], [0.5], [0.2], [8.3]])),
                q(torch.tensor([0, 1, 0, 1])),
                False,
                False,
                "macro",
                0.875,
            ]
        )
        TESTS.append(
            [
                p(torch.tensor([0.5, 0.5, 0.2, 8.3])),
                q(torch.tensor([0, 1, 0, 1])),
                False,
                False,
                "macro",
                0.875,
            ]
        )
        TESTS.append(
            [
                p(torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5]])),
                q(torch.tensor([[0], [1], [0], [1]])),
                True,
                True,
                "none",
                [0.75, 0.75],
            ]
        )
        TESTS.append(
            [
                p(torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5], [0.1, 0.5]])),
                q(torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1], [0, 1]])),
                True,
                False,
                "weighted",
                0.56667,
            ]
        )
        TESTS.append(
            [
                p(torch.tensor([[0.1, 0.9], [0.3, 1.4], [0.2, 0.1], [0.1, 0.5], [0.1, 0.5]])),
                q(torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1], [0, 1]])),
                True,
                False,
                "micro",
                0.62,
            ]
        )


class TestComputeROCAUC(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_value(self, y_pred, y, softmax, to_onehot, average, expected_value):
        y_pred_trans = Compose([ToTensor(), Activations(softmax=softmax)])
        y_trans = Compose([ToTensor(), AsDiscrete(to_onehot=to_onehot, n_classes=2)])
        y_pred = torch.stack([y_pred_trans(i) for i in decollate_batch(y_pred)], dim=0)
        y = torch.stack([y_trans(i) for i in decollate_batch(y)], dim=0)
        result = compute_roc_auc(y_pred=y_pred, y=y, average=average)
        np.testing.assert_allclose(expected_value, result, rtol=1e-5)

    @parameterized.expand(TESTS)
    def test_class_value(self, y_pred, y, softmax, to_onehot, average, expected_value):
        y_pred_trans = Compose([ToTensor(), Activations(softmax=softmax)])
        y_trans = Compose([ToTensor(), AsDiscrete(to_onehot=to_onehot, n_classes=2)])
        y_pred = [y_pred_trans(i) for i in decollate_batch(y_pred)]
        y = [y_trans(i) for i in decollate_batch(y)]
        metric = ROCAUCMetric(average=average)
        metric(y_pred=y_pred, y=y)
        result = metric.aggregate()
        metric.reset()
        np.testing.assert_allclose(expected_value, result, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
