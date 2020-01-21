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
        data_key = 'image'
        normalizer = IntensityNormalizer(data_key)  # test a single key
        normalised = normalizer({data_key: self.imt})
        expected = (self.imt - np.mean(self.imt)) / np.std(self.imt)
        self.assertTrue(np.allclose(normalised[data_key], expected))

    def test_image_normalizer_default_1(self):
        data_key = 'image'
        normalizer = IntensityNormalizer([data_key])  # test list of keys
        normalised = normalizer({data_key: self.imt})
        expected = (self.imt - np.mean(self.imt)) / np.std(self.imt)
        self.assertTrue(np.allclose(normalised[data_key], expected))

    def test_image_normalizer_default_2(self):
        data_keys = ['image_1', 'image_2']
        normalizer = IntensityNormalizer(data_keys)  # test list of keys
        normalised = normalizer(dict(zip(data_keys, (self.imt, self.seg1))))
        expected_1 = (self.imt - np.mean(self.imt)) / np.std(self.imt)
        expected_2 = (self.seg1 - np.mean(self.seg1)) / np.std(self.seg1)
        self.assertTrue(np.allclose(normalised[data_keys[0]], expected_1))
        self.assertTrue(np.allclose(normalised[data_keys[1]], expected_2))


if __name__ == '__main__':
    unittest.main()
