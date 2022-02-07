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
from monai.transforms.utility.dictionary import RandLambdad


class RandTest(Randomizable):
    """
    randomisable transform for testing.
    """

    def randomize(self, data=None):
        self._a = self.R.random()

    def __call__(self, data):
        self.randomize()
        return data + self._a


class TestRandLambdad(unittest.TestCase):
    def test_rand_lambdad_identity(self):
        img = np.zeros((10, 10))
        data = {"img": img, "prop": 1.0}

        test_func = RandTest()
        test_func.set_random_state(seed=134)
        expected = {"img": test_func(data["img"]), "prop": 1.0}
        test_func.set_random_state(seed=134)
        ret = RandLambdad(keys=["img", "prop"], func=test_func, overwrite=[True, False])(data)
        np.testing.assert_allclose(expected["img"], ret["img"])
        np.testing.assert_allclose(expected["prop"], ret["prop"])
        ret = RandLambdad(keys=["img", "prop"], func=test_func, prob=0.0)(data)
        np.testing.assert_allclose(data["img"], ret["img"])
        np.testing.assert_allclose(data["prop"], ret["prop"])

        trans = RandLambdad(keys=["img", "prop"], func=test_func, prob=0.5)
        trans.set_random_state(seed=123)
        ret = trans(data)
        np.testing.assert_allclose(data["img"], ret["img"])
        np.testing.assert_allclose(data["prop"], ret["prop"])


if __name__ == "__main__":
    unittest.main()
