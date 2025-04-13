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

from monai.networks.utils import pixelshuffle, pixelunshuffle


class TestPixelUnshuffle(unittest.TestCase):

    def test_2d_basic(self):
        x = torch.randn(2, 4, 16, 16)
        out = pixelunshuffle(x, spatial_dims=2, scale_factor=2)
        self.assertEqual(out.shape, (2, 16, 8, 8))

    def test_3d_basic(self):
        x = torch.randn(2, 4, 16, 16, 16)
        out = pixelunshuffle(x, spatial_dims=3, scale_factor=2)
        self.assertEqual(out.shape, (2, 32, 8, 8, 8))

    def test_non_square_input(self):
        x = torch.arange(192).reshape(1, 2, 12, 8)
        out = pixelunshuffle(x, spatial_dims=2, scale_factor=2)
        torch.testing.assert_close(out, torch.pixel_unshuffle(x, 2))

    def test_different_scale_factor(self):
        x = torch.arange(360).reshape(1, 2, 12, 15)
        out = pixelunshuffle(x, spatial_dims=2, scale_factor=3)
        torch.testing.assert_close(out, torch.pixel_unshuffle(x, 3))

    def test_inverse_operation(self):
        x = torch.arange(4096).reshape(1, 8, 8, 8, 8)
        shuffled = pixelshuffle(x, spatial_dims=3, scale_factor=2)
        unshuffled = pixelunshuffle(shuffled, spatial_dims=3, scale_factor=2)
        torch.testing.assert_close(x, unshuffled)


if __name__ == "__main__":
    unittest.main()
