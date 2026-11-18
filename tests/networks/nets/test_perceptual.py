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
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from parameterized import parameterized
from torch import nn

from monai.losses import PerceptualLoss
from monai.networks.nets import LPIPS, LPIPSFeatures, NetLinLayer, ScalingLayer
from monai.utils import optional_import
from tests.test_utils import dict_product, skip_if_quick

torchvision, has_torchvision = optional_import("torchvision")


TEST_NETWORKS = [("alex", "alexnet"), ("vgg", "vgg16"), ("squeeze", "squeezenet1_1")]
TEST_PRETRAINED = [(params,) for params in dict_product(network=TEST_NETWORKS, pretrained=[True, False])]


class TestLPIPSHelpers(unittest.TestCase):
    """Check public helpers and the sequential feature data flow."""

    def test_feature_slices_are_chained(self):
        """Later slices must consume preceding features rather than the RGB input."""
        first = nn.Conv2d(3, 4, 1, bias=False)
        second = nn.Conv2d(4, 5, 1, bias=False)
        with torch.no_grad():
            first.weight.fill_(1.0)
            second.weight.fill_(0.5)
        model = LPIPSFeatures(nn.Sequential(first, nn.ReLU(), second), (2, 3))
        self.assertIsInstance(model, nn.Sequential)
        outputs = model(torch.ones(1, 3, 2, 2))
        torch.testing.assert_close(outputs[0], torch.full((1, 4, 2, 2), 3.0))
        torch.testing.assert_close(outputs[1], torch.full((1, 5, 2, 2), 6.0))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_sequential_calibration_preserves_checkpoint_key(self):
        """The Sequential implementation retains upstream model.1.weight keys."""
        model = NetLinLayer(3).eval()
        self.assertIsInstance(model, nn.Sequential)
        with torch.no_grad():
            model.model[1].weight.fill_(1.0)
        torch.testing.assert_close(model(torch.ones(1, 3, 2, 2)), torch.full((1, 1, 2, 2), 3.0))
        self.assertEqual(set(model.state_dict()), {"model.1.weight"})

    def test_grayscale_scaling(self):
        """Grayscale input uses the original three channel scaling constants."""
        output = ScalingLayer()(torch.zeros(1, 1, 2, 2))
        expected = torch.tensor([0.0655021834, 0.1964285714, 0.4177777778]).view(1, 3, 1, 1)
        torch.testing.assert_close(output, expected.expand(1, 3, 2, 2))


@unittest.skipUnless(has_torchvision, "Requires torchvision")
@skip_if_quick
class TestLPIPSPretrained(unittest.TestCase):
    @parameterized.expand(TEST_PRETRAINED)
    def test_calibration_checkpoint(self, params):
        net, factory_name = params["network"]
        factory = getattr(torchvision.models, factory_name)
        expected = {"lin0.model.1.weight": torch.full((1, 64, 1, 1), 0.125)}
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / f"lpips_v0.1_{net}.pth"

            def save_checkpoint(url, filepath, **kwargs):
                self.assertEqual(filepath, checkpoint)
                torch.save(expected, filepath)

            with (
                patch("monai.networks.nets.perceptual.LPIPS_CHECKPOINTS_DIR", Path(directory)),
                patch("monai.networks.nets.perceptual.download_url", side_effect=save_checkpoint),
                patch.object(torchvision.models, factory_name, side_effect=lambda **kwargs: factory(weights=None)),
            ):
                model = LPIPS(net=net, pretrained=params["pretrained"])
            self.assertEqual(checkpoint.is_file(), params["pretrained"])
            self.assertEqual(
                torch.equal(model.state_dict()["lin0.model.1.weight"], expected["lin0.model.1.weight"]),
                params["pretrained"],
            )

    def test_perceptual_loss_cache_dir(self):
        factory = torchvision.models.squeezenet1_1
        previous_cache = torch.hub.get_dir()
        self.addCleanup(torch.hub.set_dir, previous_cache)
        expected = {"lin0.model.1.weight": torch.full((1, 64, 1, 1), 0.125)}
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory, "checkpoints", "lpips_v0.1_squeeze.pth")

            def save_checkpoint(url, filepath, **kwargs):
                self.assertEqual(filepath, checkpoint)
                filepath.parent.mkdir(parents=True, exist_ok=True)
                torch.save(expected, filepath)

            with (
                patch("monai.networks.nets.perceptual.download_url", side_effect=save_checkpoint),
                patch.object(torchvision.models, "squeezenet1_1", side_effect=lambda **kwargs: factory(weights=None)),
                self.assertWarnsRegex(UserWarning, "Setting cache_dir"),
            ):
                loss = PerceptualLoss(spatial_dims=2, network_type="squeeze", cache_dir=directory)
            self.assertTrue(checkpoint.is_file())
            torch.testing.assert_close(
                loss.perceptual_function.state_dict()["lin0.model.1.weight"], expected["lin0.model.1.weight"]
            )


if __name__ == "__main__":
    unittest.main()
