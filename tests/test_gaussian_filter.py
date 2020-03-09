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

from monai.networks.layers.simplelayers import GaussianFilter


class GaussianFilterTestCase(unittest.TestCase):

    def test_1d(self):
        a = torch.ones(1, 8, 10)
        g = GaussianFilter(1, 3, 3, torch.device('cpu:0'))
        expected = np.array([[
            [
                0.56658804, 0.69108766, 0.79392236, 0.86594427, 0.90267116, 0.9026711, 0.8659443, 0.7939224, 0.6910876,
                0.56658804
            ],
        ]])
        expected = np.tile(expected, (1, 8, 1))
        np.testing.assert_allclose(g(a).cpu().numpy(), expected)

    def test_2d(self):
        a = torch.ones(1, 1, 3, 3)
        g = GaussianFilter(2, 3, 3, torch.device('cpu:0'))
        expected = np.array([[[[0.13380532, 0.14087981, 0.13380532], [0.14087981, 0.14832835, 0.14087981],
                               [0.13380532, 0.14087981, 0.13380532]]]])

        np.testing.assert_allclose(g(a).cpu().numpy(), expected)

    def test_3d(self):
        a = torch.ones(1, 1, 4, 3, 4)
        g = GaussianFilter(3, 3, 3, torch.device('cpu:0'))
        expected = np.array(
            [[[[[0.07294822, 0.08033235, 0.08033235, 0.07294822], [0.07680509, 0.08457965, 0.08457965, 0.07680509],
                [0.07294822, 0.08033235, 0.08033235, 0.07294822]],
               [[0.08033235, 0.08846395, 0.08846395, 0.08033235], [0.08457965, 0.09314119, 0.09314119, 0.08457966],
                [0.08033235, 0.08846396, 0.08846396, 0.08033236]],
               [[0.08033235, 0.08846395, 0.08846395, 0.08033235], [0.08457965, 0.09314119, 0.09314119, 0.08457966],
                [0.08033235, 0.08846396, 0.08846396, 0.08033236]],
               [[0.07294822, 0.08033235, 0.08033235, 0.07294822], [0.07680509, 0.08457965, 0.08457965, 0.07680509],
                [0.07294822, 0.08033235, 0.08033235, 0.07294822]]]]],)
        np.testing.assert_allclose(g(a).cpu().numpy(), expected)


if __name__ == '__main__':
    unittest.main()
