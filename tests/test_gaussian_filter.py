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

from monai.networks.layers import GaussianFilter


class GaussianFilterTestCase(unittest.TestCase):
    def test_1d(self):
        a = torch.ones(1, 8, 10)
        g = GaussianFilter(1, 3, 3).to(torch.device("cpu:0"))
        expected = np.array(
            [
                [
                    [
                        0.5654129,
                        0.68915915,
                        0.79146194,
                        0.8631974,
                        0.8998163,
                        0.8998163,
                        0.8631973,
                        0.79146194,
                        0.6891592,
                        0.5654129,
                    ]
                ]
            ]
        )
        expected = np.tile(expected, (1, 8, 1))
        np.testing.assert_allclose(g(a).cpu().numpy(), expected, rtol=1e-5)

    def test_2d(self):
        a = torch.ones(1, 1, 3, 3)
        g = GaussianFilter(2, 3, 3).to(torch.device("cpu:0"))
        expected = np.array(
            [
                [
                    [
                        [0.13239081, 0.13932934, 0.13239081],
                        [0.13932936, 0.14663152, 0.13932936],
                        [0.13239081, 0.13932934, 0.13239081],
                    ]
                ]
            ]
        )

        np.testing.assert_allclose(g(a).cpu().numpy(), expected, rtol=1e-5)
        if torch.cuda.is_available():
            g = GaussianFilter(2, 3, 3).to(torch.device("cuda:0"))
            np.testing.assert_allclose(g(a.cuda()).cpu().numpy(), expected, rtol=1e-2)

    def test_3d(self):
        a = torch.ones(1, 1, 4, 3, 4)
        g = GaussianFilter(3, 3, 3).to(torch.device("cpu:0"))

        expected = np.array(
            [
                [
                    [
                        [
                            [0.07189433, 0.07911152, 0.07911152, 0.07189433],
                            [0.07566228, 0.08325771, 0.08325771, 0.07566228],
                            [0.07189433, 0.07911152, 0.07911152, 0.07189433],
                        ],
                        [
                            [0.07911152, 0.08705322, 0.08705322, 0.07911152],
                            [0.08325771, 0.09161563, 0.09161563, 0.08325771],
                            [0.07911152, 0.08705322, 0.08705322, 0.07911152],
                        ],
                        [
                            [0.07911152, 0.08705322, 0.08705322, 0.07911152],
                            [0.08325771, 0.09161563, 0.09161563, 0.08325771],
                            [0.07911152, 0.08705322, 0.08705322, 0.07911152],
                        ],
                        [
                            [0.07189433, 0.07911152, 0.07911152, 0.07189433],
                            [0.07566228, 0.08325771, 0.08325771, 0.07566228],
                            [0.07189433, 0.07911152, 0.07911152, 0.07189433],
                        ],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(g(a).cpu().numpy(), expected, rtol=1e-5)

    def test_3d_sigmas(self):
        a = torch.ones(1, 1, 4, 3, 2)
        g = GaussianFilter(3, [3, 2, 1], 3).to(torch.device("cpu:0"))

        expected = np.array(
            [
                [
                    [
                        [[0.13690521, 0.13690521], [0.15181276, 0.15181276], [0.13690521, 0.13690521]],
                        [[0.1506486, 0.15064861], [0.16705267, 0.16705267], [0.1506486, 0.15064861]],
                        [[0.1506486, 0.15064861], [0.16705267, 0.16705267], [0.1506486, 0.15064861]],
                        [[0.13690521, 0.13690521], [0.15181276, 0.15181276], [0.13690521, 0.13690521]],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(g(a).cpu().numpy(), expected, rtol=1e-5)
        if torch.cuda.is_available():
            g = GaussianFilter(3, [3, 2, 1], 3).to(torch.device("cuda:0"))
            np.testing.assert_allclose(g(a.cuda()).cpu().numpy(), expected, rtol=1e-2)

    def test_wrong_args(self):
        with self.assertRaisesRegex(ValueError, ""):
            GaussianFilter(3, [3, 2], 3).to(torch.device("cpu:0"))
        GaussianFilter(3, [3, 2, 1], 3).to(torch.device("cpu:0"))  # test init


if __name__ == "__main__":
    unittest.main()
