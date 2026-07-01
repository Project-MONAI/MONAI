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
from parameterized import parameterized

import monai.transforms as mt
from monai.data import MetaTensor, set_track_meta
from monai.transforms.lazy.functional import apply_pending
from monai.transforms.lazy.utils import kwargs_from_pending
from monai.utils import LazyAttr, TraceKeys

LABEL_MODE_CASES = [
    ["spacing", mt.Spacing, {"pixdim": (1.5, 1.5, 1.5), "mode": "nearest", "padding_mode": "zeros"}],
    ["rotate", mt.Rotate, {"angle": (0.0, 0.0, 0.3), "mode": "nearest", "keep_size": True}],
]


class TestLazyResampleMode(unittest.TestCase):
    def setUp(self):
        set_track_meta(True)

    @parameterized.expand(LABEL_MODE_CASES)
    def test_lazy_preserves_nearest_labels(self, _, xform_cls, xform_kwargs):
        labels = torch.zeros(1, 20, 20, 20)
        labels[0, 4:16, 4:16, 4:16] = 1
        labels[0, 8:12, 8:12, 8:12] = 2
        xform = xform_cls(**xform_kwargs)
        xform.lazy = True
        out, _ = apply_pending(xform(MetaTensor(labels)))
        self.assertEqual(set(out.unique().tolist()), {0.0, 1.0, 2.0})

    def test_kwargs_from_pending_reads_extra_info(self):
        pending = {TraceKeys.EXTRA_INFO: {"mode": "nearest", "padding_mode": "border", "align_corners": True}}
        kwargs = kwargs_from_pending(pending)
        self.assertEqual(kwargs[LazyAttr.INTERP_MODE], "nearest")
        self.assertEqual(kwargs[LazyAttr.PADDING_MODE], "border")
        self.assertTrue(kwargs[LazyAttr.ALIGN_CORNERS])

    def test_kwargs_from_pending_drops_interpolate_only_modes(self):
        for mode in ("area", "nearest-exact"):
            kwargs = kwargs_from_pending({TraceKeys.EXTRA_INFO: {"mode": mode}})
            self.assertIsNone(kwargs[LazyAttr.INTERP_MODE])


if __name__ == "__main__":
    unittest.main()
