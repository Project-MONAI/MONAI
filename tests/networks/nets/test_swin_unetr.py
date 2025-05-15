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

import os
import tempfile
import unittest
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.apps import download_url
from monai.networks import eval_mode
from monai.networks.nets.swin_unetr import PatchMerging, PatchMergingV2, SwinUNETR, filter_swinunetr
from monai.networks.utils import copy_model_state
from monai.utils import optional_import
from tests.test_utils import dict_product
from tests.test_utils import (
    assert_allclose,
    skip_if_downloading_fails,
    skip_if_no_cuda,
    skip_if_quick,
    testing_data_config,
)

einops, has_einops = optional_import("einops")

test_merging_mode = ["mergingv2", "merging", PatchMerging, PatchMergingV2]
checkpoint_vals = [True, False]

TEST_CASE_SWIN_UNETR = [
    [
        {
            "spatial_dims": len(params["img_size"]),
            "in_channels": params["in_channels"],
            "out_channels": params["out_channels"],
            "img_size": params["img_size"],
            "feature_size": params["feature_size"],
            "depths": params["depth"],
            "norm_name": params["norm_name"],
            "attn_drop_rate": params["attn_drop_rate"],
            "downsample": test_merging_mode[i % len(test_merging_mode)],
            "use_checkpoint": params["use_checkpoint"],
        },
        (2, params["in_channels"], *params["img_size"]),
        (2, params["out_channels"], *params["img_size"]),
    ]
    for i, params in enumerate(
        dict_product(
            attn_drop_rate=[0.4],
            in_channels=[1],
            depth=[[2, 1, 1, 1], [1, 2, 1, 1]],
            out_channels=[2],
            img_size=((64, 32, 192), (96, 32)),
            feature_size=[12],
            norm_name=["instance"],
            use_checkpoint=checkpoint_vals,
        )
    )
]

TEST_CASE_FILTER = [
    [
        {"in_channels": 1, "out_channels": 14, "feature_size": 48, "use_checkpoint": True},
        "swinViT.layers1.0.blocks.0.norm1.weight",
        torch.tensor([0.9473, 0.9343, 0.8566, 0.8487, 0.8065, 0.7779, 0.6333, 0.5555]),
    ]
]


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
            SwinUNETR(spatial_dims=1, in_channels=1, out_channels=2, feature_size=48, norm_name="instance")

        with self.assertRaises(ValueError):
            SwinUNETR(in_channels=1, out_channels=4, feature_size=50, norm_name="instance")

        with self.assertRaises(ValueError):
            SwinUNETR(in_channels=1, out_channels=3, feature_size=24, norm_name="instance", drop_rate=-1)

    def test_patch_merging(self):
        dim = 10
        t = PatchMerging(dim)(torch.zeros((1, 21, 20, 20, dim)))
        self.assertEqual(t.shape, torch.Size([1, 11, 10, 10, 20]))

    @parameterized.expand(TEST_CASE_FILTER)
    @skip_if_quick
    @skip_if_no_cuda
    def test_filter_swinunetr(self, input_param, key, value):
        with skip_if_downloading_fails():
            with tempfile.TemporaryDirectory() as tempdir:
                file_name = "ssl_pretrained_weights.pth"
                data_spec = testing_data_config("models", f"{file_name.split('.', 1)[0]}")
                weight_path = os.path.join(tempdir, file_name)
                download_url(
                    data_spec["url"], weight_path, hash_val=data_spec["hash_val"], hash_type=data_spec["hash_type"]
                )

                ssl_weight = torch.load(weight_path, weights_only=True)["model"]
                net = SwinUNETR(**input_param)
                dst_dict, loaded, not_loaded = copy_model_state(net, ssl_weight, filter_func=filter_swinunetr)
                assert_allclose(dst_dict[key][:8], value, atol=1e-4, rtol=1e-4, type_test=False)
                self.assertTrue(len(loaded) == 157 and len(not_loaded) == 2)


if __name__ == "__main__":
    unittest.main()
