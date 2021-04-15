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

from parameterized import parameterized

from monai.apps.pathology.transforms import ExtractStainsMacenko
from monai.utils import exact_version, optional_import

cp, has_cp = optional_import("cupy", "8.6.0", exact_version)

# input pixels are all transparent and below the beta absorbance threshold
EXTRACT_STAINS_TEST_CASE_1 = [
    cp.zeros((3, 2, 3)),
    cp.array([[0.0, 0.0], [0.70710678, 0.70710678], [0.70710678, 0.70710678]]),
]

# input pixels are all the same, but above beta absorbance threshold
EXTRACT_STAINS_TEST_CASE_2 = [
    cp.full((3, 2, 3), 200),
    cp.array([[0.57735027, 0.57735027], [0.57735027, 0.57735027], [0.57735027, 0.57735027]]),
]

# input pixels are all transparent and below the beta absorbance threshold
NORMALIZE_STAINS_TEST_CASE_1 = [
    {},
    cp.zeros((3, 2, 3)),
    cp.array([[[63, 25, 60], [63, 25, 60]], [[63, 25, 60], [63, 25, 60]], [[63, 25, 60], [63, 25, 60]]]),
]

# input pixels are all the same, but above beta absorbance threshold
NORMALIZE_STAINS_TEST_CASE_2 = [
    {},
    cp.full((3, 2, 3), 200),
    cp.array([[[63, 25, 60], [63, 25, 60]], [[63, 25, 60], [63, 25, 60]], [[63, 25, 60], [63, 25, 60]]]),
]

# with a custom target_he, which is the same as the image's stain matrix
NORMALIZE_STAINS_TEST_CASE_3 = [
    {"target_he": cp.full((3, 2), 0.57735027)},
    cp.full((3, 2, 3), 200),
    cp.full((3, 2, 3), 42),
]


class TestExtractStainsMacenko(unittest.TestCase):
    @parameterized.expand([EXTRACT_STAINS_TEST_CASE_1, EXTRACT_STAINS_TEST_CASE_2])
    @unittests.skipUnless(has_cp, "Requires CuPy")
    def test_value(self, image, expected_data):
        result = ExtractStainsMacenko()(image)
        cp.testing.assert_allclose(result, expected_data)


class TestNormalizeStainsMacenko(unittest.TestCase):
    @parameterized.expand([NORMALIZE_STAINS_TEST_CASE_1, NORMALIZE_STAINS_TEST_CASE_2, NORMALIZE_STAINS_TEST_CASE_3])
    @unittests.skipUnless(has_cp, "Requires CuPy")
    def test_value(self, argments, image, expected_data):
        result = NormalizeStainsMacenko(**argments)(image)
        cp.testing.assert_allclose(result, expected_data)


if __name__ == "__main__":
    unittest.main()
