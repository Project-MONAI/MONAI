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

from monai.apps.pathology.transforms import PostProcessHoVerNetOutput
from monai.transforms.intensity.array import ComputeHoVerMaps
from monai.networks.nets import HoVerNet
from monai.utils import min_version, optional_import
from tests.utils import TEST_NDARRAYS

_, has_skimage = optional_import("skimage", "0.19.3", min_version)


TESTS = []
for p in TEST_NDARRAYS:
    TEST_CASE_MASK = np.zeros((1, 10, 10), dtype="int16")
    TEST_CASE_MASK[:, 2:6, 2:6] = 1
    TEST_CASE_MASK[:, 7:10, 7:10] = 2
    mask = p(TEST_CASE_MASK)

    TEST_CASE_NP = np.zeros((1, 10, 10))
    TEST_CASE_NP[:, 2:6, 2:6] = 0.7
    TEST_CASE_NP[:, 7:10, 7:10] = 0.6
    probs_map_np = p(TEST_CASE_NP)

    TEST_CASE_NC = np.zeros((2, 10, 10))
    TEST_CASE_NC[0, 2:6, 2:6] = 0.8
    TEST_CASE_NC[1, 2:6, 2:6] = 1.8
    TEST_CASE_NC[0, 7:10, 7:10] = 0.6
    TEST_CASE_NC[1, 7:10, 7:10] = 1.9
    probs_map_nc = p(TEST_CASE_NC)

    expected_shape = (1, 10, 10)
    TESTS.append(
        [
            {
                "threshold_pred": 0.5,
                "threshold_overall": 0.4,
                "min_size": 4,
                "sigma": 0.4,
                "kernel_size": 3,
                "radius": 2,
            },
            p,
            probs_map_np,
            probs_map_nc,
            mask,
            expected_shape,
        ]
    )


@unittest.skipUnless(has_skimage, "Requires scikit-image library.")
class TestGetInstanceLevelSegMap(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_output(self, args, in_type, probs_map_np, probs_map_nc, mask, expected):

        hover_map = in_type(ComputeHoVerMaps()(mask))
        pred = {
            HoVerNet.Branch.NP.value: probs_map_np,
            HoVerNet.Branch.NC.value: probs_map_nc,
            HoVerNet.Branch.HV.value: hover_map,
        }

        postprocesshovernetoutput = PostProcessHoVerNetOutput(**args)
        output = postprocesshovernetoutput(pred)

        # temporarily only test shape
        self.assertTupleEqual(output[0].shape, expected)


if __name__ == "__main__":
    unittest.main()
