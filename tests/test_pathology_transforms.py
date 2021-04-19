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

# input pixels all transparent and below the beta absorbance threshold
EXTRACT_STAINS_TEST_CASE_1 = [
    cp.full((3, 2, 3), 240),
]

# input pixels uniformly filled, but above beta absorbance threshold
EXTRACT_STAINS_TEST_CASE_2 = [
    cp.full((3, 2, 3), 100),
]

# input pixels uniformly filled (different value), but above beta absorbance threshold
EXTRACT_STAINS_TEST_CASE_3 = [
    cp.full((3, 2, 3), 150),
]

# input pixels uniformly filled with zeros, leading to two identical stains extracted
EXTRACT_STAINS_TEST_CASE_4 = [
    cp.zeros((3, 2, 3)),
    cp.array([[0.0, 0.0], [0.70710678, 0.70710678], [0.70710678, 0.70710678]]),
]

# input pixels not uniformly filled, leading to two different stains extracted
EXTRACT_STAINS_TEST_CASE_5 = [
    cp.array([[[100, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]),
    cp.array([[0.70710677, 0.18696113], [0.0, 0.0], [0.70710677, 0.98236734]]),
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
    @parameterized.expand([EXTRACT_STAINS_TEST_CASE_1])
    @unittest.skipUnless(has_cp, "Requires CuPy")
    def test_transparent_image(self, image):
        """
        Test Macenko stain extraction on an image that comprises
        only transparent pixels - pixels with absorbance below the
        beta absorbance threshold. A ValueError should be raised,
        since once the transparent pixels are removed, there are no
        remaining pixels to compute eigenvectors.
        """
        with self.assertRaises(ValueError):
            ExtractStainsMacenko()(image)

    @parameterized.expand([EXTRACT_STAINS_TEST_CASE_2, EXTRACT_STAINS_TEST_CASE_3])
    @unittest.skipUnless(has_cp, "Requires CuPy")
    def test_identical_result_vectors(self, image):
        """
        Test Macenko stain extraction on input images that are
        uniformly filled with pixels that have absorbance above the
        beta absorbance threshold. Since input image is uniformly filled,
        the two extracted stains should have the same RGB values. So,
        we assert that the first column is equal to the second column
        of the returned stain matrix.
        """
        result = ExtractStainsMacenko()(image)
        cp.testing.assert_array_equal(result[:, 0], result[:, 1])

    @parameterized.expand([EXTRACT_STAINS_TEST_CASE_4, EXTRACT_STAINS_TEST_CASE_5])
    @unittest.skipUnless(has_cp, "Requires CuPy")
    def test_result_value(self, image, expected_data):
        """
        Test that an input image returns an expected stain matrix.

        For test case 4:
        - a uniformly filled input image should result in
          eigenvectors [[1,0,0],[0,1,0],[0,0,1]]
        - phi should be an array containing only values of
          arctan(1) since the ratio between the eigenvectors
          corresponding to the two largest eigenvalues is 1
        - maximum phi and minimum phi should thus be arctan(1)
        - thus, maximum vector and minimum vector should be
          [[0],[0.70710677],[0.70710677]]
        - the resulting extracted stain should be
          [[0,0],[0.70710678,0.70710678],[0.70710678,0.70710678]]

        For test case 5:
        - the non-uniformly filled input image should result in
          eigenvectors [[0,0,1],[1,0,0],[0,1,0]]
        - maximum phi and minimum phi should thus be 0.785 and
          0.188 respectively
        - thus, maximum vector and minimum vector should be
          [[0.18696113],[0],[0.98236734]] and
          [[0.70710677],[0],[0.70710677]] respectively
        - the resulting extracted stain should be
          [[0.70710677,0.18696113],[0,0],[0.70710677,0.98236734]]
        """
        result = ExtractStainsMacenko()(image)
        cp.testing.assert_allclose(result, expected_data)


class TestNormalizeStainsMacenko(unittest.TestCase):
    @parameterized.expand([NORMALIZE_STAINS_TEST_CASE_1, NORMALIZE_STAINS_TEST_CASE_2, NORMALIZE_STAINS_TEST_CASE_3])
    @unittest.skipUnless(has_cp, "Requires CuPy")
    def test_value(self, argments, image, expected_data):
        result = NormalizeStainsMacenko(**argments)(image)
        cp.testing.assert_allclose(result, expected_data)


if __name__ == "__main__":
    unittest.main()
