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
import torch.nn as nn

from monai.inferers import MCDropoutInferer, SimpleInferer, SlidingWindowInferer


def _net(with_dropout: bool = True) -> nn.Module:
    layers = [nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()]
    if with_dropout:
        layers.append(nn.Dropout2d(0.5))
    layers.append(nn.Conv2d(4, 2, 1))
    return nn.Sequential(*layers).eval()


class TestMCDropoutInferer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.inputs = torch.rand(2, 1, 16, 16)

    def test_samples_vary_and_have_leading_sample_dim(self):
        net = _net()
        samples = MCDropoutInferer(num_samples=5)(self.inputs, net)
        self.assertEqual(samples.shape, (5, 2, 2, 16, 16))
        self.assertFalse(torch.allclose(samples[0], samples[1]))

    def test_modes_are_restored_and_batchnorm_stays_in_eval(self):
        net = _net()
        bn = net[1]
        running_mean = bn.running_mean.clone()
        MCDropoutInferer(num_samples=3)(self.inputs, net)
        self.assertFalse(any(m.training for m in net.modules()))
        torch.testing.assert_close(bn.running_mean, running_mean)

    def test_modes_are_restored_on_error(self):
        net = _net()

        def failing_network(x):
            raise RuntimeError("boom")

        failing_network.modules = net.modules
        with self.assertRaises(RuntimeError):
            MCDropoutInferer(num_samples=2)(self.inputs, failing_network)
        self.assertFalse(any(m.training for m in net.modules()))

    def test_return_mean(self):
        net = _net()
        torch.manual_seed(1)
        samples = MCDropoutInferer(num_samples=4)(self.inputs, net)
        torch.manual_seed(1)
        mean = MCDropoutInferer(num_samples=4, return_samples=False)(self.inputs, net)
        torch.testing.assert_close(mean, samples.mean(dim=0))

    def test_wraps_sliding_window_inferer(self):
        net = _net()
        inferer = MCDropoutInferer(SlidingWindowInferer(roi_size=(8, 8), sw_batch_size=2), num_samples=3)
        samples = inferer(self.inputs, net)
        self.assertEqual(samples.shape, (3, 2, 2, 16, 16))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            MCDropoutInferer(num_samples=0)
        with self.assertRaises(ValueError):
            MCDropoutInferer(SimpleInferer())(self.inputs, _net(with_dropout=False))


if __name__ == "__main__":
    unittest.main()
