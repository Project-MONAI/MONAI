# Copyright 2020 MONAI Consortium
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
from monai.transforms.transforms import ImageEndPadder

TEST_CASE_1 = [
    {
        'out_size': [16, 16, 8],
        'mode': 'constant'
    },
    np.zeros((1, 3, 8, 8, 4)),
    np.zeros((1, 3, 16, 16, 8)),
]

class TestImageEndPadder(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1])
    def test_image_end_pad_shape(self, input_param, input_data, expected_val):
        padder = ImageEndPadder(**input_param)
        result = padder(input_data)
        self.assertAlmostEqual(result.shape, expected_val.shape)


if __name__ == '__main__':
    unittest.main()
