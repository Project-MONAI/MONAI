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

from __future__ import annotations

import unittest

import torch
from parameterized import parameterized

from monai.apps.reconstruction.networks.nets.coil_sensitivity_model import CoilSensitivityModel, _fully_sampled_mask
from tests.test_utils import skip_if_no_cuda, test_export_save, test_script_save

FEATURES = (2, 2, 2, 2, 2, 2)

# number of samples, then the sampled positions as (start, stop), then the expected region as (start, stop)
TEST_REGIONS = [
    [20, [(8, 12)], (8, 12)],
    [21, [(8, 13)], (8, 13)],
    [32, [(12, 20), (2, 4), (28, 30)], (12, 20)],
    [32, [(10, 20)], (11, 21)],  # off-center regions are recentered, as they have always been
    [32, [(0, 32)], (0, 32)],
    [32, [(0, 20)], (6, 26)],
    [32, [(2, 4)], (0, 0)],
    [32, [(16, 32)], (8, 24)],  # used to raise IndexError as the region reaches the last index
    [32, [(0, 20), (30, 32)], (6, 26)],  # used to wrap past index 0 into the samples at the end
]


def _mask(num_samples, sampled):
    mask = torch.zeros(1, 1, 1, num_samples, 1, dtype=torch.bool)
    for start, stop in sampled:
        mask[..., start:stop, :] = True
    return mask


class TestCoilSensitivityModel(unittest.TestCase):
    @parameterized.expand(TEST_REGIONS)
    def test_fully_sampled_mask(self, num_samples, sampled, expected):
        expected_mask = _mask(num_samples, [expected])
        result = _fully_sampled_mask(_mask(num_samples, sampled))
        self.assertTrue(torch.equal(result, expected_mask))

    @parameterized.expand(
        [[20, [(8, 12)]], [21, [(8, 13)]], [32, [(12, 20), (2, 4), (28, 30)]], [32, [(10, 20)]], [32, [(2, 4)]]]
    )
    def test_matches_get_fully_sampled_region(self, num_samples, sampled):
        model = CoilSensitivityModel(spatial_dims=2)
        mask = _mask(num_samples, sampled)
        left, right = model.get_fully_sampled_region(mask)
        start = (num_samples - (right - left) + 1) // 2
        expected = torch.zeros_like(mask)
        expected[..., start : start + right - left, :] = True
        self.assertTrue(torch.equal(_fully_sampled_mask(mask), expected))

    def test_multiple_sampling_vectors(self):
        mask = torch.ones(2, 1, 1, 32, 1, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "single sampling vector"):
            _fully_sampled_mask(mask)

    def test_values_outside_region_are_dropped(self):
        net = CoilSensitivityModel(spatial_dims=2, features=FEATURES).eval()
        data = torch.randn(1, 3, 32, 32, 2)
        data[..., 2, :] = float("inf")
        with torch.no_grad():
            self.assertTrue(torch.isfinite(net(data, _mask(32, [(12, 20)]))).all())

    @skip_if_no_cuda
    def test_exported_program_moves_to_gpu(self):
        net = CoilSensitivityModel(spatial_dims=2, features=FEATURES).eval()
        data, mask = torch.randn(1, 3, 32, 32, 2), _mask(32, [(12, 20)])
        exported = torch.export.export(net, args=(data, mask)).module().to("cuda:0")
        self.assertEqual(exported(data.to("cuda:0"), mask.to("cuda:0")).device.type, "cuda")

    def test_script(self):
        net = CoilSensitivityModel(spatial_dims=2, features=FEATURES)
        test_script_save(net, torch.randn(1, 3, 32, 32, 2), _mask(32, [(12, 20)]))

    def test_export(self):
        net = CoilSensitivityModel(spatial_dims=2, features=FEATURES)
        test_export_save(net, torch.randn(1, 3, 32, 32, 2), _mask(32, [(12, 20)]))

    def test_export_follows_mask(self):
        net = CoilSensitivityModel(spatial_dims=2, features=FEATURES).eval()
        data = torch.randn(1, 3, 32, 32, 2)
        exported = torch.export.export(net, args=(data, _mask(32, [(12, 20)]))).module()
        other_mask = _mask(32, [(10, 22)])
        torch.testing.assert_close(exported(data, other_mask), net(data, other_mask))


if __name__ == "__main__":
    unittest.main()
