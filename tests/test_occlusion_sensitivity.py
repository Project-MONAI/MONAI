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

from monai.networks.nets import DenseNet, DenseNet121
from monai.visualize import OcclusionSensitivity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_channels_2d = 4
out_channels_3d = 3
model_2d = DenseNet121(spatial_dims=2, in_channels=1, out_channels=out_channels_2d).to(device)
model_2d_2c = DenseNet121(spatial_dims=2, in_channels=2, out_channels=out_channels_2d).to(device)
model_3d = DenseNet(
    spatial_dims=3, in_channels=1, out_channels=out_channels_3d, init_features=2, growth_rate=2, block_config=(6,)
).to(device)
model_2d.eval()
model_2d_2c.eval()
model_3d.eval()

# 2D w/ bounding box
TEST_CASE_0 = [
    {"nn_module": model_2d},
    {"x": torch.rand(1, 1, 48, 64).to(device), "b_box": [-1, -1, 2, 40, 1, 62]},
    (1, 1, 39, 62, out_channels_2d),
    (1, 1, 39, 62),
]
# 3D w/ bounding box and stride
TEST_CASE_1 = [
    {"nn_module": model_3d, "n_batch": 10, "stride": (2, 1, 2), "mask_size": (16, 15, 14)},
    {"x": torch.rand(1, 1, 6, 6, 6).to(device), "b_box": [-1, -1, 2, 3, -1, -1, -1, -1]},
    (1, 1, 2, 6, 6, out_channels_3d),
    (1, 1, 2, 6, 6),
]

TEST_CASE_FAIL_0 = [  # 2D should fail, since 3 stride values given
    {"nn_module": model_2d, "n_batch": 10, "stride": (2, 2, 2)},
    {"x": torch.rand(1, 1, 48, 64).to(device), "b_box": [-1, -1, 2, 3, -1, -1]},
]

TEST_CASE_FAIL_1 = [  # 2D should fail, since stride is not a factor of image size
    {"nn_module": model_2d, "stride": 3},
    {"x": torch.rand(1, 1, 48, 64).to(device)},
]
TEST_MULTI_CHANNEL = [
    {"nn_module": model_2d_2c, "per_channel": False},
    {"x": torch.rand(1, 2, 48, 64).to(device)},
    (1, 1, 48, 64, out_channels_2d),
    (1, 1, 48, 64),
]


class TestComputeOcclusionSensitivity(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_MULTI_CHANNEL])
    def test_shape(self, init_data, call_data, map_expected_shape, most_prob_expected_shape):
        occ_sens = OcclusionSensitivity(**init_data)
        m, most_prob = occ_sens(**call_data)
        self.assertTupleEqual(m.shape, map_expected_shape)
        self.assertTupleEqual(most_prob.shape, most_prob_expected_shape)
        # most probable class should be of type int, and should have min>=0, max<num_classes
        self.assertEqual(most_prob.dtype, torch.int64)
        self.assertGreaterEqual(most_prob.min(), 0)
        self.assertLess(most_prob.max(), m.shape[-1])

    @parameterized.expand([TEST_CASE_FAIL_0, TEST_CASE_FAIL_1])
    def test_fail(self, init_data, call_data):
        occ_sens = OcclusionSensitivity(**init_data)
        with self.assertRaises(ValueError):
            occ_sens(**call_data)


if __name__ == "__main__":
    unittest.main()
