# Copyright (c) MONAI Consortium
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

from monai.transforms.transform import Randomizable
from monai.transforms.utility.array import RandLambda


class RandTest(Randomizable):
    """
    randomisable transform for testing.
    """

    def randomize(self, data=None):
        self._a = self.R.random()

    def __call__(self, data):
        self.randomize()
        return data + self._a


class TestRandLambda(unittest.TestCase):
    def test_rand_lambdad_identity(self):
        img = np.zeros((10, 10))

        test_func = RandTest()
        test_func.set_random_state(seed=134)
        expected = test_func(img)
        test_func.set_random_state(seed=134)
        ret = RandLambda(func=test_func)(img)
        np.testing.assert_allclose(expected, ret)
        ret = RandLambda(func=test_func, prob=0.0)(img)
        np.testing.assert_allclose(img, ret)

        trans = RandLambda(func=test_func, prob=0.5)
        trans.set_random_state(seed=123)
        ret = trans(img)
        np.testing.assert_allclose(img, ret)


if __name__ == "__main__":
    unittest.main()
