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
from parameterized import parameterized

from monai.transforms import RandFlipd
from tests.utils import NumpyImageTestCase2D

VALID_CASES = [("no_axis", None), ("one_axis", 1), ("many_axis", [0, 1])]


class TestRandFlipd(NumpyImageTestCase2D):
    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, _, spatial_axis):
        flip = RandFlipd(keys="img", prob=1.0, spatial_axis=spatial_axis)
        res = flip({"img": self.imt[0]})
        expected = []
        for channel in self.imt[0]:
            expected.append(np.flip(channel, spatial_axis))
        expected = np.stack(expected)
        self.assertTrue(np.allclose(expected, res["img"]))


if __name__ == "__main__":
    unittest.main()
