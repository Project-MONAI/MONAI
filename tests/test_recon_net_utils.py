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

import torch
from parameterized import parameterized
from tests.utils import assert_allclose
from monai.apps.reconstruction.networks.nets.utils import (
    reshape_channel_complex_to_last_dim,
    complex_normalize,
    reshape_complex_to_channel_dim,
    pad,
    inverse_complex_normalize,
)


# no need for checking devices, these functions don't change device format
# reshape test case
im_2d, im_3d = torch.ones([3,4,50,70,2]), torch.ones([3,4,50,70,80,2])
TEST_RESHAPE = [(im_2d,), (im_3d,)]

# normalize test case
im_2d, im_3d = torch.randint(0,3,[3,4,50,70]).float(), torch.randint(0,3,[3,4,50,70,80]).float()
TEST_NORMALIZE = [(im_2d,), (im_3d,)]

# pad test case
im_2d, im_3d = torch.ones([3,4,50,70]), torch.ones([3,4,50,70,80])
TEST_PAD = [(im_2d,), (im_3d,)]

class TestReconNetUtils(unittest.TestCase):
    @parameterized.expand(TEST_RESHAPE)
    def test_reshape_channel_complex(self, test_data):
        result = reshape_complex_to_channel_dim(test_data)
        result = reshape_channel_complex_to_last_dim(result)
        self.assertEqual(result.shape, test_data.shape)

    @parameterized.expand(TEST_NORMALIZE)
    def test_complex_normalize(self, test_data):
        result, mean, std = complex_normalize(test_data)
        result = inverse_complex_normalize(result, mean, std)
        self.assertTrue( (((result-test_data)**2).mean()**0.5).item() < 1e-5 )

    @parameterized.expand(TEST_PAD)
    def test_pad(self, test_data):
        result, padder = pad(test_data)
        result = padder.inverse(result)
        assert_allclose(result, test_data)


if __name__ == "__main__":
    unittest.main()
