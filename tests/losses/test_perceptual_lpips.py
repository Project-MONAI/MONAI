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
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from parameterized import parameterized

from monai.losses import PerceptualLoss
from monai.networks.nets import LPIPS
from monai.utils import optional_import
from tests.lpips_test_utils import lpips_test_cache
from tests.test_utils import skip_if_quick

torchvision, has_torchvision = optional_import("torchvision")


class TestPerceptualLPIPSValidation(unittest.TestCase):
    """Check argument validation before any optional backbone is constructed."""

    def test_unsupported_network(self):
        """Reject unsupported names before constructing a Torchvision backbone."""
        with self.assertRaisesRegex(ValueError, "Unsupported LPIPS network"):
            LPIPS(net="unsupported")


@unittest.skipUnless(has_torchvision, "Requires torchvision")
@skip_if_quick
class TestPerceptualLPIPS(unittest.TestCase):
    def setUp(self):
        """Use local calibration fixtures without replacing cache/hash/load logic."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        _, self.calibrations, _ = stack.enter_context(lpips_test_cache())

    @parameterized.expand(
        [
            (net, factory, weights, pretrained)
            for net, factory, weights in (
                ("alex", "alexnet", "AlexNet_Weights"),
                ("vgg", "vgg16", "VGG16_Weights"),
                ("squeeze", "squeezenet1_1", "SqueezeNet1_1_Weights"),
            )
            for pretrained in (True, False)
        ]
    )
    def test_distance_and_gradient_without_lpips(self, net, factory_name, weights_name, pretrained):
        # Only replace external weight downloading; use the actual Torchvision networks.
        factory = getattr(torchvision.models, factory_name)
        torch.manual_seed(9284)
        with patch.object(
            torchvision.models, factory_name, side_effect=lambda **kwargs: factory(weights=None)
        ) as create_backbone:
            loss = PerceptualLoss(spatial_dims=2, network_type=net, pretrained=pretrained)
        create_backbone.assert_called_once_with(weights=getattr(torchvision.models, weights_name).IMAGENET1K_V1)

        model = loss.perceptual_function
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.net.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.lins.parameters()))
        calibrated_state = self.calibrations[net]
        for key, value in calibrated_state.items():
            self.assertEqual(torch.equal(model.state_dict()[key], value), pretrained)
        for channels in (1, 3):
            with self.subTest(channels=channels):
                source = torch.rand(2, channels, 64, 64, requires_grad=True)
                target = torch.rand_like(source)
                self.assertEqual(loss(source, source).item(), 0.0)
                distance = model(source, target)
                self.assertEqual(distance.shape, (2, 1, 1, 1))
                self.assertTrue(torch.isfinite(distance).all())
                if pretrained:
                    self.assertTrue(torch.all(distance > 0))
                else:
                    # Random calibration layers need not define a nonnegative metric.
                    self.assertTrue(torch.all(distance != 0))
                torch.testing.assert_close(distance, model(target, source))
                torch.testing.assert_close(loss(source, target), distance.mean())
                distance.mean().backward()
                self.assertTrue(torch.isfinite(source.grad).all())
                self.assertGreater(source.grad.abs().sum().item(), 0)

        # Existing checkpoints contain both names for the same registered calibration layers.
        state = model.state_dict()
        self.assertIn("net.slice1.0.weight", state)
        torch.testing.assert_close(state["lin0.model.1.weight"], state["lins.0.model.1.weight"])
        self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))

    @parameterized.expand([("alex", "alexnet"), ("vgg", "vgg16"), ("squeeze", "squeezenet1_1")])
    def test_load_legacy_state_dict(self, net, factory_name):
        # Captured from unmodified LPIPS v0.1, independent of MONAI's module registration.
        manifest_path = Path(__file__).parents[1] / "testing_data" / "lpips_v0_1_state_dict.json"
        manifest = json.loads(manifest_path.read_text())["networks"][net]
        factory = getattr(torchvision.models, factory_name)
        with patch.object(torchvision.models, factory_name, side_effect=lambda **kwargs: factory(weights=None)):
            loss = PerceptualLoss(spatial_dims=2, network_type=net)
        legacy_state = {f"perceptual_function.{key}": torch.full(shape, 0.125) for key, shape in manifest.items()}
        loss.load_state_dict(legacy_state, strict=True)
        self.assertEqual(set(loss.state_dict()), set(legacy_state))
        for key, value in loss.state_dict().items():
            with self.subTest(key=key):
                self.assertEqual(tuple(value.shape), tuple(manifest[key.removeprefix("perceptual_function.")]))
                torch.testing.assert_close(value, legacy_state[key], rtol=0, atol=0)

    def test_fake_3d_matches_three_axes(self):
        factory = torchvision.models.squeezenet1_1
        with patch.object(torchvision.models, "squeezenet1_1", side_effect=lambda **kwargs: factory(weights=None)):
            loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", fake_3d_ratio=1.0)
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
