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
import torch
from parameterized import parameterized

from monai.data import MetaTensor, set_track_meta
from monai.transforms import RandFlipd
from tests.utils import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, assert_allclose, test_local_inversion

VALID_CASES = [("no_axis", None), ("one_axis", 1), ("many_axis", [0, 1])]


class TestRandFlipd(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, spatial_axis):
        for p in TEST_NDARRAYS_ALL:
            flip = RandFlipd(keys="img", prob=1.0, spatial_axis=spatial_axis)
            im = p(self.imt[0])
            result = flip({"img": im})["img"]
            expected = [np.flip(channel, spatial_axis) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(result, p(expected), type_test=False)
            test_local_inversion(flip, {"img": result}, {"img": im}, "img")
            set_track_meta(False)
            result = flip({"img": im})["img"]
            self.assertNotIsInstance(result, MetaTensor)
            self.assertIsInstance(result, torch.Tensor)
            set_track_meta(True)


if __name__ == "__main__":
    unittest.main()
