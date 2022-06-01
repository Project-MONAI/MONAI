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

from monai.data import MetaTensor
from monai.transforms import RandAxisFlip
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestRandAxisFlip(NumpyImageTestCase2D):
    def test_correct_results(self):
        for p in TEST_NDARRAYS:
            flip = RandAxisFlip(prob=1.0)
            im = p(self.imt[0])
            result = flip(im)
            expected = [np.flip(channel, flip._axis) for channel in self.imt[0]]
            assert_allclose(result, p(np.stack(expected)), type_test=False)

            if isinstance(im, MetaTensor):
                im_inv = flip.inverse(result)
                assert_allclose(im_inv, p(self.imt[0]))
                assert_allclose(im_inv.affine, im.affine)


if __name__ == "__main__":
    unittest.main()
