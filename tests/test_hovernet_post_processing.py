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
from parameterized import parameterized

from monai.apps.pathology.transforms.post.array import HoVerNetPostProcessing
from monai.transforms import ComputeHoVerMaps, FillHoles, GaussianSmooth
from monai.utils import min_version, optional_import
from tests.utils import TEST_NDARRAYS, assert_allclose

_, has_scipy = optional_import("scipy", "1.8.1", min_version)
_, has_skimage = optional_import("skimage", "0.19.3", min_version)

y, x = np.ogrid[0:30, 0:30]
image = (x - 10) ** 2 + (y - 10) ** 2 <= 5**2
image = image[None, ...].astype("uint8")

TEST_CASE_1 = [{}, [{"1": [10, 10]}, np.zeros_like(image), np.zeros_like(image)]]
TEST_CASE_2 = [{"distance_smooth_fn": GaussianSmooth()}, [{"1": [10, 10]}, np.zeros_like(image), np.zeros_like(image)]]
TEST_CASE_3 = [{"marker_postprocess_fn": FillHoles()}, [{"1": [10, 10]}, np.zeros_like(image), np.zeros_like(image)]]

TEST_CASE = []
for p in TEST_NDARRAYS:
    TEST_CASE.append([p, image] + TEST_CASE_1)
    TEST_CASE.append([p, image] + TEST_CASE_2)
    TEST_CASE.append([p, image] + TEST_CASE_3)


@unittest.skipUnless(has_scipy, "Requires scipy library.")
@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
class TestHoVerNetPostProcessing(unittest.TestCase):
    @parameterized.expand(TEST_CASE)
    def test_value(self, in_type, test_data, kwargs, expected):
        nuclear_prediction = in_type(test_data.astype(float))
        hover_map = in_type(ComputeHoVerMaps()(test_data.astype(int)))
        nuclear_type = in_type(test_data)

        pred_dict, inst_map, type_map = HoVerNetPostProcessing(**kwargs)(nuclear_prediction, hover_map, nuclear_type)
        # instance prediction info
        for key in pred_dict:
            assert_allclose(pred_dict[key]["centroid"], expected[0][key], type_test=False)

        # instance map
        assert_allclose(inst_map, expected[1], type_test=False)

        # type map
        if expected[2] is None:
            self.assertIsNone(type_map)
        else:
            assert_allclose(type_map, expected[2], type_test=False)


if __name__ == "__main__":
    unittest.main()
