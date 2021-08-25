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

from monai.transforms.intensity.dictionary import ScaleIntensityRangePercentilesd
from tests.utils import NumpyImageTestCase2D


class TestScaleIntensityRangePercentilesd(NumpyImageTestCase2D):
    def test_scaling(self):
        img = self.imt
        data = {}
        data["img"] = img
        lower = 10
        upper = 99
        b_min = 0
        b_max = 255

        a_min = np.percentile(img, lower)
        a_max = np.percentile(img, upper)
        expected = (img - a_min) / (a_max - a_min)
        expected = (expected * (b_max - b_min)) + b_min

        scaler = ScaleIntensityRangePercentilesd(keys=data.keys(), lower=lower, upper=upper, b_min=b_min, b_max=b_max)

        self.assertTrue(np.allclose(expected, scaler(data)["img"]))

    def test_relative_scaling(self):
        img = self.imt
        data = {}
        data["img"] = img
        lower = 10
        upper = 99
        b_min = 100
        b_max = 300
        scaler = ScaleIntensityRangePercentilesd(
            keys=data.keys(), lower=lower, upper=upper, b_min=b_min, b_max=b_max, relative=True
        )

        expected_a_min = np.percentile(img, lower)
        expected_a_max = np.percentile(img, upper)
        expected_b_min = ((b_max - b_min) * (lower / 100.0)) + b_min
        expected_b_max = ((b_max - b_min) * (upper / 100.0)) + b_min
        expected_img = (img - expected_a_min) / (expected_a_max - expected_a_min)
        expected_img = (expected_img * (expected_b_max - expected_b_min)) + expected_b_min

        self.assertTrue(np.allclose(expected_img, scaler(data)["img"]))

    def test_invalid_instantiation(self):
        self.assertRaises(
            ValueError, ScaleIntensityRangePercentilesd, keys=["img"], lower=-1, upper=99, b_min=0, b_max=255
        )
        self.assertRaises(
            ValueError, ScaleIntensityRangePercentilesd, keys=["img"], lower=101, upper=99, b_min=0, b_max=255
        )
        self.assertRaises(
            ValueError, ScaleIntensityRangePercentilesd, keys=["img"], lower=30, upper=-2, b_min=0, b_max=255
        )
        self.assertRaises(
            ValueError, ScaleIntensityRangePercentilesd, keys=["img"], lower=30, upper=1000, b_min=0, b_max=255
        )


if __name__ == "__main__":
    unittest.main()
