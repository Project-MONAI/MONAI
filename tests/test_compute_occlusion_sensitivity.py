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

import torch
from parameterized import parameterized

from monai.metrics import compute_occlusion_sensitivity
from monai.networks.nets import DenseNet, densenet121

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_2d = densenet121(spatial_dims=2, in_channels=1, out_channels=3).to(device)
model_3d = DenseNet(
    spatial_dims=3, in_channels=1, out_channels=3, init_features=2, growth_rate=2, block_config=(6,)
).to(device)
model_2d.eval()
model_3d.eval()

# 2D w/ bounding box
TEST_CASE_0 = [
    {
        "model": model_2d,
        "image": torch.rand(1, 1, 48, 64).to(device),
        "label": torch.tensor([[0]], dtype=torch.int64).to(device),
        "b_box": [-1, -1, 2, 40, 1, 62],
    },
    (39, 62),
]
# 3D w/ bounding box
TEST_CASE_1 = [
    {
        "model": model_3d,
        "image": torch.rand(1, 1, 6, 6, 6).to(device),
        "label": 0,
        "b_box": [-1, -1, 2, 3, -1, -1, -1, -1],
        "n_batch": 10,
        "stride": 2,
    },
    (2, 6, 6),
]


class TestComputeOcclusionSensitivity(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1])
    def test_shape(self, input_data, expected_shape):
        result = compute_occlusion_sensitivity(**input_data)
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
