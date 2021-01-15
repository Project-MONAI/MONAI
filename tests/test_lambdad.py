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

from monai.transforms.utility.dictionary import Lambdad
from tests.utils import NumpyImageTestCase2D


class TestLambdad(NumpyImageTestCase2D):
    def test_lambdad_identity(self):
        img = self.imt
        data = {}
        data["img"] = img

        def identity_func(x):
            return x

        lambd = Lambdad(keys=data.keys(), func=identity_func)
        expected = data
        expected["img"] = identity_func(data["img"])
        self.assertTrue(np.allclose(expected["img"], lambd(data)["img"]))

    def test_lambdad_slicing(self):
        img = self.imt
        data = {}
        data["img"] = img

        def slice_func(x):
            return x[:, :, :6, ::-2]

        lambd = Lambdad(keys=data.keys(), func=slice_func)
        expected = {}
        expected["img"] = slice_func(data["img"])
        self.assertTrue(np.allclose(expected["img"], lambd(data)["img"]))


if __name__ == "__main__":
    unittest.main()
