import unittest

import numpy as np

from monai.data.transforms.intensity_normalizer import IntensityNormalizer
from tests.utils import NumpyImageTestCase2D


class MyTestCase(NumpyImageTestCase2D):

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


if __name__ == '__main__':
    unittest.main()
