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

from monai.data.transforms.intensity_normalizer import IntensityNormalizer
from tests.utils import NumpyImageTestCase2D


class IntensityNormTestCase(NumpyImageTestCase2D):

    def test_image_normalizer_default(self):
        normalizer = IntensityNormalizer()
        normalised = normalizer(self.imt)
        expected = (self.imt - np.mean(self.imt)) / np.std(self.imt)
        self.assertTrue(np.allclose(normalised, expected))


if __name__ == '__main__':
    unittest.main()
