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

from monai.transforms import RandGibbsNoise
from tests.utils import TorchImageTestCase2D, TorchImageTestCase3D


class TestRandGibbsNoise(TorchImageTestCase3D, TorchImageTestCase2D):
    """
    Tests below check the extreme cases of the transform.
    It should act as identity for alpha = 0 and as the zero
    map for alpha = 1. The functions check the 2D and 3D cases.
    We also check the case when alpha is a list.
    """

    def test_correct_results(self):
        # check both 3d and 2d cases
        self.test_3d()
        self.test_2d()

    def test_3d(self):

        TorchImageTestCase3D.setUp(self)

        # check identity
        id_map = RandGibbsNoise(prob=1, alpha=0)
        new = id_map(self.imt)
        np.testing.assert_allclose(self.imt, new, atol=1e-3)

        # check zero vector
        zero_map = RandGibbsNoise(prob=1, alpha=1)
        new = zero_map(self.imt)
        np.testing.assert_allclose(0 * self.imt, new)

        # check alpha = [a,b] works
        g = RandGibbsNoise(prob=1.0, alpha=[0.50, 0.51])
        new = g(self.imt)
        self.assertTrue(
            (g.sampled_alpha >= 0.50 and g.sampled_alpha <= 0.51), "sampling alpha outside provided interval"
        )

    def test_2d(self):

        TorchImageTestCase2D.setUp(self)

        # check identity
        id_map = RandGibbsNoise(prob=1, alpha=0, dim=2)
        new = id_map(self.imt)
        np.testing.assert_allclose(self.imt, new, atol=1e-3)

        # check zero vector
        zero_map = RandGibbsNoise(prob=1, alpha=1, dim=2)
        new = zero_map(self.imt)
        np.testing.assert_allclose(0 * self.imt, new)

        # check alpha = [a,b] works
        g = RandGibbsNoise(prob=1.0, alpha=[0.50, 0.51], dim=2)
        new = g(self.imt)
        self.assertTrue(
            (g.sampled_alpha >= 0.50 and g.sampled_alpha <= 0.51), "sampling alpha outside provided interval"
        )


if __name__ == "__main__":
    unittest.main()
