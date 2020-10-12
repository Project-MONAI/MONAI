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

from monai.networks.layers.convutils import gaussian_1d


class TestGaussian1d(unittest.TestCase):
    def test_gaussian(self):
        np.testing.assert_allclose(
            gaussian_1d(0.5, 8),
            torch.tensor(
                [
                    0.0000e00,
                    2.9802e-07,
                    1.3496e-03,
                    1.5731e-01,
                    6.8269e-01,
                    1.5731e-01,
                    1.3496e-03,
                    2.9802e-07,
                    0.0000e00,
                ]
            ),
            rtol=1e-4,
        )

        np.testing.assert_allclose(
            gaussian_1d(1, 1),
            torch.tensor([0.24173, 0.382925, 0.24173]),
            rtol=1e-4,
        )

    def test_refined_gaussian(self):
        np.testing.assert_allclose(
            gaussian_1d(0.5, 8, "refined"),
            torch.tensor(
                [
                    9.9961e-05,
                    1.6044e-03,
                    1.9352e-02,
                    1.5642e-01,
                    6.4504e-01,
                    1.5642e-01,
                    1.9352e-02,
                    1.6044e-03,
                    9.9961e-05,
                ]
            ),
            rtol=1e-4,
        )

        np.testing.assert_allclose(
            gaussian_1d(1, 1, "refined"),
            torch.tensor([0.235838, 0.528323, 0.235838]),
            rtol=1e-3,
        )

        np.testing.assert_allclose(
            gaussian_1d(10, 1, "refined"),
            torch.tensor(
                [
                    0.000995,
                    0.002378,
                    0.005276,
                    0.010819,
                    0.020423,
                    0.035327,
                    0.055749,
                    0.079926,
                    0.103705,
                    0.121408,
                    0.127987,
                    0.121408,
                    0.103705,
                    0.079926,
                    0.055749,
                    0.035327,
                    0.020423,
                    0.010819,
                    0.005276,
                    0.002378,
                    0.000995,
                ]
            ),
            rtol=1e-3,
        )

    def test_wrong_sigma(self):
        with self.assertRaises(ValueError):
            gaussian_1d(-1, 10)
        with self.assertRaises(ValueError):
            gaussian_1d(1, -10)
        with self.assertRaises(NotImplementedError):
            gaussian_1d(1, 10, "wrong_arg")


if __name__ == "__main__":
    unittest.main()
