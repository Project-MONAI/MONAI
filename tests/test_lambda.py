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

from monai.transforms.utility.array import Lambda
from tests.utils import NumpyImageTestCase2D


class TestLambda(NumpyImageTestCase2D):
    def test_lambda_identity(self):
        img = self.imt
        func = lambda x: x
        lambd = Lambda(func=func)
        self.assertTrue(np.allclose(func(img), lambd(img)))

    def test_lambda_slicing(self):
        img = self.imt
        func = lambda x: x[:, :, :6, ::-2]
        lambd = Lambda(func=func)
        self.assertTrue(np.allclose(func(img), lambd(img)))


if __name__ == "__main__":
    unittest.main()
