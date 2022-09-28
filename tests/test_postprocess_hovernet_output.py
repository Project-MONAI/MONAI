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

from monai.apps.pathology.transforms.post.array import PostProcessHoVerNetOutput
from monai.utils import min_version, optional_import
from tests.utils import TEST_NDARRAYS, assert_allclose

_, has_skimage = optional_import("skimage", "0.19.3", min_version)
_, has_scipy = optional_import("scipy", "1.8.1", min_version)
Image, has_pil = optional_import("PIL", name="Image")

test_data_path = os.path.join(os.path.dirname(__file__), "testing_data", "hovernet_test_data_raw.npz")
prediction = np.load(test_data_path)

kwargs = {"threshold_overall": 0.4, "min_size": 10, "sigma": 0.4, "kernel_size": 21, "radius": 2}

TESTS = []
for p in TEST_NDARRAYS:
    type_pred = prediction["nc_map"]
    seg_pred = prediction["prob_map"]
    hover_map = prediction["hover_map"]
    expected = prediction["pred_instance"][None]

    TESTS.append([kwargs, p, type_pred, seg_pred, hover_map, expected, 20])


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
@unittest.skipUnless(has_scipy, "Requires scipy library.")
@unittest.skipUnless(has_pil, "Requires PIL library.")
class TestPostProcessHoVerNetOutput(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, in_type, type_pred, seg_pred, hover_map, expected, expected_num):

        postprocesshovernetoutput = PostProcessHoVerNetOutput(**args)
        output = postprocesshovernetoutput(in_type(seg_pred), in_type(hover_map), in_type(type_pred))
        pred_type = list(output[1].keys())

        self.assertTupleEqual(output[0].shape, expected.shape)
        self.assertEqual(len(pred_type), expected_num)
        self.assertEqual(output[1][pred_type[0]]["type"], None)
        self.assertEqual(output[1][pred_type[0]]["type_probability"], None)
        assert_allclose(output[0], expected, type_test=False)


if __name__ == "__main__":
    unittest.main()
