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
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import torch
from torch import nn

from monai.apps import HashCheckError
from monai.networks.nets import LPIPS, LPIPSFeatures, NetLinLayer, ScalingLayer
from monai.utils import optional_import
from tests.lpips_test_utils import lpips_test_cache

torchvision, has_torchvision = optional_import("torchvision")


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
class TestLPIPSDownloads(unittest.TestCase):
    """Exercise real download, integrity and cache logic with a local transport."""

    def setUp(self):
        """Isolate the cache and replace only ImageNet and HTTP downloads."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.cache, self.expected, self.transfer = stack.enter_context(lpips_test_cache(preload=False))
        factory = torchvision.models.squeezenet1_1
        stack.enter_context(
            patch.object(torchvision.models, "squeezenet1_1", side_effect=lambda **kwargs: factory(weights=None))
        )
        self.checkpoint = self.cache / "checkpoints" / "lpips_v0.1_squeeze.pth"

    def assert_calibration_loaded(self, model):
        """Check every calibrated layer against the independent fixture tensors.

        Args:
            model: LPIPS network whose pretrained calibration has been loaded.
        """
        for key, value in self.expected["squeeze"].items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_download_and_offline_cache_reuse(self):
        """A cold cache downloads once; a warm cache loads without HTTP access."""
        self.assert_calibration_loaded(LPIPS(net="squeeze"))
        self.assertTrue(self.checkpoint.exists())
        self.transfer.assert_called_once()
        self.assertEqual(
            self.transfer.call_args.args[0], "https://huggingface.co/MONAI/checkpoints/resolve/main/squeeze.pth"
        )
        self.transfer.side_effect = AssertionError("A valid cached checkpoint must not be downloaded again")
        self.assert_calibration_loaded(LPIPS(net="squeeze"))

    def test_random_calibration_does_not_download(self):
        """pretrained=False leaves calibration random and does not access its URL."""
        self.transfer.side_effect = AssertionError("Random calibration must not download weights")
        model = LPIPS(net="squeeze", pretrained=False)
        self.assertFalse(self.checkpoint.exists())
        for key, value in self.expected["squeeze"].items():
            self.assertFalse(torch.equal(model.state_dict()[key], value))

    def test_corrupt_download_is_rejected(self):
        """Failed integrity validation must not silently use random calibration."""
        self.transfer.side_effect = lambda url, filepath, progress: Path(filepath).write_bytes(b"invalid checkpoint")
        with self.assertRaises(HashCheckError):
            LPIPS(net="squeeze")
        self.assertFalse(self.checkpoint.exists())

    def test_corrupt_cache_is_rejected(self):
        """Cached weights are validated before being passed to torch.load."""
        LPIPS(net="squeeze")
        self.checkpoint.write_bytes(b"corrupt cache")
        self.transfer.side_effect = AssertionError("Corrupt existing cache must be reported")
        with self.assertRaises(HashCheckError):
            LPIPS(net="squeeze")

    def test_network_failure_is_reported(self):
        """An unavailable server must fail instead of returning an uncalibrated model."""
        self.transfer.side_effect = URLError("offline")
        with self.assertRaises(URLError):
            LPIPS(net="squeeze")
        self.assertFalse(self.checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
