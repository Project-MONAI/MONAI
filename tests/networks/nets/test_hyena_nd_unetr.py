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
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.networks.blocks.hyena import HyenaTransformerBlock, is_nvsubquadratic_available
from monai.networks.nets.hyena_nd_unetr import PAPER_VARIANTS, HyenaNDUNETR
from monai.networks.nets.swin_unetr import SwinTransformerBlock, SwinUNETR
from tests.test_utils import skip_if_no_cuda

HAS_NVSUBQ = is_nvsubquadratic_available()


PAPER_VARIANT_CASES = [
    ("HHHH", (True, True, True, True)),
    ("HAHA", (True, False, True, False)),
    ("HHAA", (True, True, False, False)),
]


def _block_type_at_stage(model, stage_idx):
    layer_attr = ["layers1", "layers2", "layers3", "layers4"][stage_idx]
    return type(getattr(model.swinViT, layer_attr)[0].blocks[0])


class TestHyenaNDUNETRConstructorContract(unittest.TestCase):
    """``HyenaNDUNETR.__init__`` enforces an explicit, non-empty ``hyena_stages``."""

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_explicit_stages_required(self):
        with self.assertRaisesRegex(ValueError, "requires `hyena_stages`"):
            HyenaNDUNETR(in_channels=1, out_channels=14, feature_size=12, hyena_stages=None)

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_wrong_length_stages_rejected(self):
        with self.assertRaisesRegex(ValueError, "length 4"):
            HyenaNDUNETR(in_channels=1, out_channels=14, feature_size=12, hyena_stages=(True, True))

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_all_false_stages_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one stage"):
            HyenaNDUNETR(
                in_channels=1,
                out_channels=14,
                feature_size=12,
                hyena_stages=(False, False, False, False),
            )

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_use_hyena_kwarg_rejected(self):
        """The subclass forces use_hyena=True; caller may not override via kwargs."""
        with self.assertRaisesRegex(TypeError, "use_hyena"):
            HyenaNDUNETR(
                in_channels=1,
                out_channels=14,
                feature_size=12,
                hyena_stages=(True, True, False, False),
                use_hyena=True,
            )

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_duplicate_hyena_stages_kwarg_rejected(self):
        with self.assertRaisesRegex(TypeError, "hyena_stages"):
            HyenaNDUNETR(
                in_channels=1,
                out_channels=14,
                feature_size=12,
                hyena_stages=(True, True, False, False),
                **{"hyena_stages": (True, False, True, False)},  # type: ignore[arg-type]
            )

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_subclass_of_swin_unetr(self):
        m = HyenaNDUNETR(
            in_channels=1, out_channels=14, feature_size=12, hyena_stages=(True, True, False, False)
        )
        self.assertIsInstance(m, SwinUNETR)
        # The forced kwargs land on the instance via SwinUNETR.__init__.
        self.assertTrue(m.use_hyena)
        self.assertEqual(m.hyena_stages, (True, True, False, False))


class TestHyenaNDUNETRFromPaperVariant(unittest.TestCase):
    """``from_paper_variant`` maps {HHHH, HAHA, HHAA} to the correct stage pattern."""

    @parameterized.expand(PAPER_VARIANT_CASES)
    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_returns_expected_stages(self, name, expected_stages):
        m = HyenaNDUNETR.from_paper_variant(name, in_channels=1, out_channels=14, feature_size=12)
        self.assertEqual(m.hyena_stages, expected_stages)
        for stage_idx, want_hyena in enumerate(expected_stages):
            block_type = _block_type_at_stage(m, stage_idx)
            if want_hyena:
                self.assertIs(block_type, HyenaTransformerBlock)
            else:
                self.assertIs(block_type, SwinTransformerBlock)

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_case_insensitive(self):
        m_upper = HyenaNDUNETR.from_paper_variant("HHAA", in_channels=1, out_channels=14, feature_size=12)
        m_lower = HyenaNDUNETR.from_paper_variant("hhaa", in_channels=1, out_channels=14, feature_size=12)
        self.assertEqual(m_upper.hyena_stages, m_lower.hyena_stages)

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_aaaa_rejected(self):
        """AAAA is plain SwinUNETR and intentionally not exposed via this constructor."""
        with self.assertRaisesRegex(ValueError, "Unknown paper variant"):
            HyenaNDUNETR.from_paper_variant("AAAA", in_channels=1, out_channels=14, feature_size=12)

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_unknown_variant_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown paper variant"):
            HyenaNDUNETR.from_paper_variant("HAAA", in_channels=1, out_channels=14, feature_size=12)

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_redundant_hyena_stages_kwarg_rejected(self):
        with self.assertRaisesRegex(ValueError, "do not also pass hyena_stages"):
            HyenaNDUNETR.from_paper_variant(
                "HHAA",
                in_channels=1,
                out_channels=14,
                feature_size=12,
                hyena_stages=(True, False, True, False),
            )

    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    def test_paper_variants_table_matches_constants(self):
        """Guard against the table in PAPER_VARIANTS drifting."""
        self.assertEqual(PAPER_VARIANTS["HHHH"], (True, True, True, True))
        self.assertEqual(PAPER_VARIANTS["HAHA"], (True, False, True, False))
        self.assertEqual(PAPER_VARIANTS["HHAA"], (True, True, False, False))


class TestHyenaNDUNETRForward(unittest.TestCase):
    """End-to-end forward over the three paper variants. CUDA required."""

    @parameterized.expand(PAPER_VARIANT_CASES)
    @skipUnless(HAS_NVSUBQ, "Requires nvsubquadratic")
    @skip_if_no_cuda
    def test_forward_shape(self, name, _stages):
        m = HyenaNDUNETR.from_paper_variant(
            name, in_channels=1, out_channels=14, feature_size=12
        ).cuda().eval()
        x = torch.randn(1, 1, 64, 64, 64, device="cuda")
        with torch.no_grad():
            out = m(x)
        self.assertEqual(out.shape, (1, 14, 64, 64, 64))


if __name__ == "__main__":
    unittest.main()
