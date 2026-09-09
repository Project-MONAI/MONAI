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
from monai.networks.blocks.hyena import HyenaTransformerBlock, is_nvsubquadratic_available
from monai.networks.nets.swin_unetr import (
    PatchMerging,
    PatchMergingV2,
    SwinTransformer,
    SwinTransformerBlock,
    SwinUNETR,
    filter_swinunetr,
)
from monai.networks.utils import copy_model_state
from monai.utils import optional_import
from tests.test_utils import (
    assert_allclose,
    dict_product,
    skip_if_downloading_fails,
    skip_if_no_cuda,
    skip_if_quick,
    testing_data_config,
)

einops, has_einops = optional_import("einops")
HAS_NVSUBQ = is_nvsubquadratic_available()
HAS_CUDA = torch.cuda.is_available()

test_merging_mode = ["mergingv2", "merging", PatchMerging, PatchMergingV2]
checkpoint_vals = [True, False]

TEST_CASE_SWIN_UNETR = [
    [
        {
            **{k: v for k, v in params.items() if k != "img_size"},
            "spatial_dims": len(params["img_size"]),
            "downsample": test_merging_mode[i % len(test_merging_mode)],
        },
        (2, params["in_channels"], *params["img_size"]),
        (2, params["out_channels"], *params["img_size"]),
    ]
    for i, params in enumerate(
        dict_product(
            attn_drop_rate=[0.4],
            depths=[[2, 1, 1, 1], [1, 2, 1, 1]],
            feature_size=[12],
            img_size=((64, 32, 192), (96, 32)),
            in_channels=[1],
            norm_name=["instance"],
            out_channels=[2],
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

    @skipUnless(has_einops, "Requires einops")
    def test_invalid_input_shape(self):
        # spatial dims not divisible by patch_size**5 (default patch_size=2, so must be divisible by 32)
        net = SwinUNETR(in_channels=1, out_channels=2, feature_size=24, spatial_dims=3)
        with self.assertRaises(ValueError):
            net(torch.randn(1, 1, 33, 64, 64))  # 33 is not divisible by 32

        net_2d = SwinUNETR(in_channels=1, out_channels=2, feature_size=24, spatial_dims=2)
        with self.assertRaises(ValueError):
            net_2d(torch.randn(1, 1, 48, 33))  # 33 is not divisible by 32

    @skipUnless(has_einops, "Requires einops")
    def test_flash_attention(self):
        input_param = {"in_channels": 1, "out_channels": 2, "feature_size": 12, "spatial_dims": 3}
        net_ref = SwinUNETR(use_flash_attention=False, **input_param).double()
        net_flash = SwinUNETR(use_flash_attention=True, **input_param).double()
        net_flash.load_state_dict(net_ref.state_dict())
        x = torch.randn(1, 1, 64, 64, 64, dtype=torch.float64)
        with eval_mode(net_ref, net_flash):
            ref = net_ref.swinViT(x, net_ref.normalize)
            out = net_flash.swinViT(x, net_flash.normalize)
        for a, b in zip(ref, out, strict=True):
            assert_allclose(a, b, atol=1e-6, rtol=1e-6, type_test=False)

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


# Backward-compat reference for SwinUNETR(use_hyena=False), feature_size=12, img_size=64^3,
# seeds (model=0, input=1), CPU.  Captured before the HyenaND port; the default code path must
# keep reproducing this within tolerance.  Tolerance-based (not a byte hash) so it tolerates
# benign cross-platform float drift while still catching a real change to the non-Hyena path.
HYENA_BACKCOMPAT_REF = torch.tensor(
    [
        -0.069162,
        -0.209673,
        0.543457,
        -0.111868,
        0.474825,
        0.031108,
        0.191482,
        -0.167401,
        0.091668,
        0.272223,
        -0.084950,
        -0.042126,
    ]
)


def _build_hyena_unetr(use_hyena=False, hyena_stages=None, feature_size=12, out_channels=14):
    return SwinUNETR(
        in_channels=1,
        out_channels=out_channels,
        feature_size=feature_size,
        use_hyena=use_hyena,
        hyena_stages=hyena_stages,
    )


def _block_type_at_stage(model, stage_idx):
    layer_attr = ["layers1", "layers2", "layers3", "layers4"][stage_idx]
    return type(getattr(model.swinViT, layer_attr)[0].blocks[0])


HYENA_VARIANT_CASES = [
    ("AAAA", False, None),
    ("HHHH", True, None),
    ("HAHA", True, (True, False, True, False)),
    ("HHAA", True, (True, True, False, False)),
]


class TestSwinUNETRHyenaBackCompat(unittest.TestCase):
    """The non-Hyena code path must keep reproducing its pre-port output (within tolerance)."""

    @skipUnless(has_einops, "Requires einops")
    def test_default_path_unchanged(self):
        """SwinUNETR with no hyena kwargs reproduces the pre-port reference output.

        Runs on CPU so it executes in environments without a GPU and is stable across
        platforms; ``assert_close`` tolerates benign float drift while still flagging a real
        change to the default (non-Hyena) code path.
        """
        torch.manual_seed(0)
        net = SwinUNETR(in_channels=1, out_channels=14, feature_size=12).eval()
        torch.manual_seed(1)
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            out = net(x)
        self.assertEqual(out.shape, (1, 14, 64, 64, 64))
        assert_allclose(
            out.flatten()[: HYENA_BACKCOMPAT_REF.numel()], HYENA_BACKCOMPAT_REF, atol=1e-4, rtol=1e-4, type_test=False
        )


class TestSwinUNETRHyenaStages(unittest.TestCase):
    """``hyena_stages`` must place :class:`HyenaTransformerBlock` at flagged stages and
    :class:`SwinTransformerBlock` everywhere else.  Construction-only; no CUDA required."""

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_haha_pattern(self):
        m = _build_hyena_unetr(use_hyena=True, hyena_stages=(True, False, True, False))
        self.assertIs(_block_type_at_stage(m, 0), HyenaTransformerBlock)
        self.assertIs(_block_type_at_stage(m, 1), SwinTransformerBlock)
        self.assertIs(_block_type_at_stage(m, 2), HyenaTransformerBlock)
        self.assertIs(_block_type_at_stage(m, 3), SwinTransformerBlock)

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_hhaa_pattern(self):
        m = _build_hyena_unetr(use_hyena=True, hyena_stages=(True, True, False, False))
        self.assertIs(_block_type_at_stage(m, 0), HyenaTransformerBlock)
        self.assertIs(_block_type_at_stage(m, 1), HyenaTransformerBlock)
        self.assertIs(_block_type_at_stage(m, 2), SwinTransformerBlock)
        self.assertIs(_block_type_at_stage(m, 3), SwinTransformerBlock)

    def test_aaaa_pattern_default(self):
        m = _build_hyena_unetr(use_hyena=False)
        for i in range(4):
            self.assertIs(_block_type_at_stage(m, i), SwinTransformerBlock)

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_hhhh_pattern_default(self):
        m = _build_hyena_unetr(use_hyena=True)
        for i in range(4):
            self.assertIs(_block_type_at_stage(m, i), HyenaTransformerBlock)

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_wrong_length_hyena_stages_raises(self):
        with self.assertRaisesRegex(ValueError, "hyena_stages must have length"):
            _build_hyena_unetr(use_hyena=True, hyena_stages=(True, True))


class TestSwinUNETRHyenaForward(unittest.TestCase):
    """Forward shape across the four paper variants. CUDA required."""

    @parameterized.expand(HYENA_VARIANT_CASES)
    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    @skip_if_no_cuda
    def test_forward_shape(self, _name, use_hyena, hyena_stages):
        m = _build_hyena_unetr(use_hyena=use_hyena, hyena_stages=hyena_stages).cuda()
        x = torch.randn(1, 1, 64, 64, 64, device="cuda")
        with torch.no_grad():
            out = m(x)
        self.assertEqual(out.shape, (1, 14, 64, 64, 64))


class TestSwinUNETRHyenaGradient(unittest.TestCase):
    """Backward through the HHAA variant must produce grads on at least 90 percent of params."""

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    @skip_if_no_cuda
    def test_hhaa_backward(self):
        m = _build_hyena_unetr(use_hyena=True, hyena_stages=(True, True, False, False)).cuda()
        x = torch.randn(1, 1, 64, 64, 64, device="cuda")
        m(x).sum().backward()
        total = list(m.parameters())
        with_grad = [p for p in total if p.grad is not None]
        coverage = len(with_grad) / len(total)
        self.assertGreater(coverage, 0.9, f"only {coverage:.1%} of params received gradients")


class TestSwinTransformerRoPEDivisibility(unittest.TestCase):
    """3D Hyena requires embed_dim * 2^layer % 6 == 0; 2D requires % 4."""

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_3d_rejects_non_divisible_embed_dim(self):
        with self.assertRaisesRegex(ValueError, "divisible by 6"):
            SwinTransformer(
                in_chans=1,
                embed_dim=14,
                window_size=(2, 2, 2),
                patch_size=(2, 2, 2),
                depths=(2, 2, 2, 2),
                num_heads=(3, 6, 12, 24),
                spatial_dims=3,
                use_hyena=True,
            )

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_2d_rejects_non_divisible_embed_dim(self):
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            SwinTransformer(
                in_chans=1,
                embed_dim=14,
                window_size=(2, 2),
                patch_size=(2, 2),
                depths=(2, 2, 2, 2),
                num_heads=(3, 6, 12, 24),
                spatial_dims=2,
                use_hyena=True,
            )

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_per_stage_skips_check_for_attention_stages(self):
        """Per-stage False suppresses the check for that stage; remaining Hyena stages still fire."""
        with self.assertRaisesRegex(ValueError, "divisible by 6"):
            SwinTransformer(
                in_chans=1,
                embed_dim=14,
                window_size=(2, 2, 2),
                patch_size=(2, 2, 2),
                depths=(2, 2, 2, 2),
                num_heads=(3, 6, 12, 24),
                spatial_dims=3,
                use_hyena=True,
                hyena_stages=(False, True, False, False),
            )


class TestSwinUNETRHyenaSlidingWindow(unittest.TestCase):
    """The production inference path: sliding-window inference over HHAA must succeed."""

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    @skip_if_no_cuda
    def test_swi_hhaa(self):
        from monai.inferers import sliding_window_inference

        m = _build_hyena_unetr(use_hyena=True, hyena_stages=(True, True, False, False)).cuda().eval()
        x = torch.randn(1, 1, 96, 96, 96, device="cuda")
        with torch.no_grad():
            out = sliding_window_inference(inputs=x, roi_size=(64, 64, 64), sw_batch_size=2, predictor=m, overlap=0.25)
        self.assertEqual(out.shape, (1, 14, 96, 96, 96))


if __name__ == "__main__":
    unittest.main()
