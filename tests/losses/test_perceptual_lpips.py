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

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from parameterized import parameterized

from monai.losses import PerceptualLoss
from monai.networks.nets import LPIPS
from monai.utils import optional_import
from tests.test_utils import dict_product, skip_if_quick

torchvision, has_torchvision = optional_import("torchvision")


TEST_NETWORKS = [("alex", "alexnet"), ("vgg", "vgg16"), ("squeeze", "squeezenet1_1")]
TEST_INPUTS = [(params,) for params in dict_product(network=TEST_NETWORKS, channels=[1, 3])]


class TestPerceptualLPIPSValidation(unittest.TestCase):
    def test_unsupported_network(self):
        with self.assertRaisesRegex(ValueError, "Unsupported LPIPS network"):
            LPIPS(net="unsupported")


@unittest.skipUnless(has_torchvision, "Requires torchvision")
@skip_if_quick
class TestPerceptualLPIPS(unittest.TestCase):
    def setUp(self):
        # Use real feature networks, replacing only external ImageNet downloads.
        torch.manual_seed(9284)
        for _, factory_name in TEST_NETWORKS:
            factory = getattr(torchvision.models, factory_name)
            patcher = patch.object(
                torchvision.models, factory_name, side_effect=lambda factory=factory, **kwargs: factory(weights=None)
            )
            self.addCleanup(patcher.stop)
            patcher.start()

    @parameterized.expand(TEST_INPUTS)
    def test_distance(self, params):
        loss = PerceptualLoss(spatial_dims=2, network_type=params["network"][0], pretrained=False)
        source = torch.rand(2, params["channels"], 64, 64)
        target = torch.rand_like(source)
        distance = loss.perceptual_function(source, target)
        self.assertEqual(distance.shape, (2, 1, 1, 1))
        self.assertTrue(torch.isfinite(distance).all())
        self.assertEqual(loss(source, source).item(), 0.0)
        torch.testing.assert_close(distance, loss.perceptual_function(target, source))
        torch.testing.assert_close(loss(source, target), distance.mean())

    @parameterized.expand(TEST_INPUTS)
    def test_input_gradient(self, params):
        loss = PerceptualLoss(spatial_dims=2, network_type=params["network"][0], pretrained=False)
        source = torch.rand(2, params["channels"], 64, 64, requires_grad=True)
        loss(source, torch.rand_like(source)).backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(source.grad.abs().sum().item(), 0)

    @parameterized.expand(TEST_NETWORKS)
    def test_load_legacy_state_dict(self, net, factory_name):
        # Independent key/shape manifest captured from unmodified LPIPS v0.1.
        manifest_path = Path(__file__).parents[1] / "testing_data" / "lpips_v0_1_state_dict.json"
        manifest = json.loads(manifest_path.read_text())["networks"][net]
        loss = PerceptualLoss(spatial_dims=2, network_type=net, pretrained=False)
        legacy_state = {f"perceptual_function.{key}": torch.full(shape, 0.125) for key, shape in manifest.items()}
        loss.load_state_dict(legacy_state, strict=True)
        for key, value in loss.state_dict().items():
            torch.testing.assert_close(value, legacy_state[key], rtol=0, atol=0)

    def test_fake_3d_matches_three_axes(self):
        loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", fake_3d_ratio=1.0, pretrained=False)
        source = torch.rand(1, 1, 32, 32, 32, requires_grad=True)
        target = torch.rand_like(source)
        expected = sum(
            loss.perceptual_function(
                source.movedim(axis, 1).reshape(-1, 1, 32, 32), target.movedim(axis, 1).reshape(-1, 1, 32, 32)
            ).mean()
            for axis in (2, 3, 4)
        )
        actual = loss(source, target)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(source.grad).all())


if __name__ == "__main__":
    unittest.main()
