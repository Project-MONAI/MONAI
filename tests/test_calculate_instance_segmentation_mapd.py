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

import os
import unittest

import numpy as np
from parameterized import parameterized

from monai.apps.pathology.transforms.post.dictionary import CalcualteInstanceSegmentationMapd
from monai.utils import min_version, optional_import
from tests.utils import TEST_NDARRAYS, assert_allclose

_, has_skimage = optional_import("skimage", "0.19.3", min_version)
_, has_scipy = optional_import("scipy", "1.8.1", min_version)
_, has_cv2 = optional_import("cv2")

test_data_path = os.path.join(os.path.dirname(__file__), "testing_data", "hovernet_test_data.npz")

prediction = np.load(test_data_path)

TESTS = []
for p in TEST_NDARRAYS:
    prob_map = prediction["prob_map"][None]
    hover_maph = prediction["hover_map_h"][None]
    hover_mapv = prediction["hover_map_v"][None]
    expected = prediction["pred_instance"][None]
    hover_map = np.concatenate([hover_maph, hover_mapv])

    TESTS.append(
        [
            {
                "threshold_pred": 0.5,
                "threshold_overall": 0.4,
                "min_size": 10,
                "sigma": 0.4,
                "kernel_size": 21,
                "radius": 2,
            },
            p,
            prob_map,
            hover_map,
            expected,
        ]
    )

ERROR_TESTS = []
for p in TEST_NDARRAYS:
    prob_map1 = np.zeros([2, 64, 64])
    hover_map1 = np.zeros([2, 64, 64])

    prob_map2 = np.zeros([1, 1, 64, 64])
    hover_map2 = np.zeros([2, 64, 64])

    prob_map3 = np.zeros([1, 64, 64])
    hover_map3 = np.zeros([2, 1, 64, 64])

    params = {
        "threshold_pred": 0.5,
        "threshold_overall": 0.4,
        "min_size": 10,
        "sigma": 0.4,
        "kernel_size": 21,
        "radius": 2,
    }

    ERROR_TESTS.append([params, p, prob_map1, hover_map1])
    ERROR_TESTS.append([params, p, prob_map2, hover_map2])
    ERROR_TESTS.append([params, p, prob_map3, hover_map3])


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
@unittest.skipUnless(has_scipy, "Requires scipy library.")
@unittest.skipUnless(has_cv2, "OpenCV required.")
class TestCalcualteInstanceSegmentationMapd(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, in_type, probs_map, hover_map, expected):
        data = {"image": in_type(probs_map), "hover": in_type(hover_map)}
        output = CalcualteInstanceSegmentationMapd(keys="image", hover_key="hover", **args)(data)

        self.assertTupleEqual(output["image"].shape, expected.shape)
        assert_allclose(output["image"], expected, type_test=False)

    @parameterized.expand(ERROR_TESTS)
    def test_value_error(self, args, in_type, probs_map, hover_map):
        data = {"image": in_type(probs_map), "hover": in_type(hover_map)}
        calculate_instance_seg = CalcualteInstanceSegmentationMapd(keys="image", hover_key="hover", **args)
        with self.assertRaises(ValueError):
            calculate_instance_seg(data)


if __name__ == "__main__":
    unittest.main()
