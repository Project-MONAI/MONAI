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

import os
import numpy as np
from parameterized import parameterized

from monai.apps.pathology.transforms.post.array import CalcualteInstanceSegmentationMap
from monai.transforms.intensity.array import ComputeHoVerMaps
from monai.utils import min_version, optional_import
from tests.utils import TEST_NDARRAYS, assert_allclose

_, has_skimage = optional_import("skimage", "0.19.3", min_version)
_, has_scipy = optional_import("scipy", "1.8.1", min_version)
Image,  has_pil= optional_import("PIL", name="Image")

input_image_path = os.path.join(os.path.dirname(__file__), "testing_data", "hovernet_input_image.png")
test_data_path = os.path.join(os.path.dirname(__file__), "testing_data", "hovernet_test_data.npz")

test_image = np.asarray(Image.open(input_image_path))
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


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
@unittest.skipUnless(has_scipy, "Requires scipy library.")
@unittest.skipUnless(has_pil, "Requires PIL library.")
class TestCalcualteInstanceSegmentationMap(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, in_type, probs_map, hover_map, expected):

        caculate_instance_seg = CalcualteInstanceSegmentationMap(**args)
        output = caculate_instance_seg(in_type(probs_map), in_type(hover_map))

        self.assertTupleEqual(output.shape, expected.shape)
        assert_allclose(output, expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
