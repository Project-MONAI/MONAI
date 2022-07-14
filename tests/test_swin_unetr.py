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
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.swin_unetr import PatchMerging, SwinUNETR
from monai.utils import optional_import

einops, has_einops = optional_import("einops")

TEST_CASE_SWIN_UNETR = []
for attn_drop_rate in [0.4]:
    for in_channels in [1]:
        for depth in [[2, 1, 1, 1], [1, 2, 1, 1]]:
            for out_channels in [2]:
                for img_size in ((64, 32, 192), (96, 32)):
                    for feature_size in [12]:
                        for norm_name in ["instance"]:
                            test_case = [
                                {
                                    "spatial_dims": len(img_size),
                                    "in_channels": in_channels,
                                    "out_channels": out_channels,
                                    "img_size": img_size,
                                    "feature_size": feature_size,
                                    "depths": depth,
                                    "norm_name": norm_name,
                                    "attn_drop_rate": attn_drop_rate,
                                },
                                (2, in_channels, *img_size),
                                (2, out_channels, *img_size),
                            ]
                            TEST_CASE_SWIN_UNETR.append(test_case)


class TestSWINUNETR(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SWIN_UNETR)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SwinUNETR(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SwinUNETR(
                in_channels=1,
                out_channels=3,
                img_size=(128, 128, 128),
                feature_size=24,
                norm_name="instance",
                attn_drop_rate=4,
            )

        with self.assertRaises(ValueError):
            SwinUNETR(in_channels=1, out_channels=2, img_size=(96, 96), feature_size=48, norm_name="instance")

        with self.assertRaises(ValueError):
            SwinUNETR(in_channels=1, out_channels=4, img_size=(96, 96, 96), feature_size=50, norm_name="instance")

        with self.assertRaises(ValueError):
            SwinUNETR(
                in_channels=1,
                out_channels=3,
                img_size=(85, 85, 85),
                feature_size=24,
                norm_name="instance",
                drop_rate=0.4,
            )

    def test_patch_merging(self):
        dim = 10
        t = PatchMerging(dim)(torch.zeros((1, 21, 20, 20, dim)))
        self.assertEqual(t.shape, torch.Size([1, 11, 10, 10, 20]))


if __name__ == "__main__":
    unittest.main()
