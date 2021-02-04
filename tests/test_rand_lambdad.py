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

from monai.transforms import Randomizable
from monai.transforms.utility.dictionary import RandLambdad
from tests.utils import NumpyImageTestCase2D


class RandTest(Randomizable):
    """
    randomisable transform for testing.
    """

    def randomize(self, data=None):
        self.set_random_state(seed=134)
        self._a = self.R.random()
        self.set_random_state(seed=None)

    def __call__(self, data):
        self.randomize()
        return data + self._a


class TestRandLambdad(NumpyImageTestCase2D):
    def test_rand_lambdad_identity(self):
        img = self.imt
        data = {"img": img, "prop": 1.0}

        test_func = RandTest()

        expected = {"img": test_func(data["img"]), "prop": 1.0}
        ret = RandLambdad(keys=["img", "prop"], func=test_func, overwrite=[True, False])(data)
        self.assertTrue(np.allclose(expected["img"], ret["img"]))
        self.assertTrue(np.allclose(expected["prop"], ret["prop"]))


if __name__ == "__main__":
    unittest.main()
