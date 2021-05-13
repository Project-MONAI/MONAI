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

from monai.transforms import GibbsNoised
from tests.utils import TorchImageTestCase2D, TorchImageTestCase3D


class TestGibbsNoised(TorchImageTestCase3D, TorchImageTestCase2D):
    """
       Test below checks the extreme cases of the transform. 
       It should act as identity for alpha = 0 and as the zero
       map for alpha = 1. The functions check the 2D and 3D cases,
       for both the image and the label.
     """

    def test_correct_results(self):

        self.test_3d()
        self.test_2d()

    def test_3d(self):

        TorchImageTestCase3D.setUp(self)
        data = {"image": self.imt, "label": self.segn}

        # check identity
        id_map = GibbsNoised(["image", "label"], alpha=0)
        data_new = id_map(data)
        np.testing.assert_allclose(self.imt, data_new["image"], atol=1e-3)
        np.testing.assert_allclose(self.segn, data_new["label"], atol=1e-2)

        # check zero mapping
        zero_map = GibbsNoised(["image", "label"], alpha=1)
        data_new = zero_map(data)
        np.testing.assert_allclose(0 * self.imt, data_new["image"])
        np.testing.assert_allclose(0 * self.segn, data_new["label"])

    def test_2d(self):

        TorchImageTestCase2D.setUp(self)
        data = {"image": self.imt, "label": self.segn}

        # check identity
        id_map = GibbsNoised(["image", "label"], alpha=0, dim=2)
        data_new = id_map(data)
        np.testing.assert_allclose(self.imt, data_new["image"], atol=1e-3)
        np.testing.assert_allclose(self.segn, data_new["label"], atol=1e-2)

        # check zero mapping
        zero_map = GibbsNoised(["image", "label"], alpha=1, dim=2)
        data_new = zero_map(data)
        np.testing.assert_allclose(0 * self.imt, data_new["image"])
        np.testing.assert_allclose(0 * self.segn, data_new["label"])


if __name__ == "__main__":
    unittest.main()
