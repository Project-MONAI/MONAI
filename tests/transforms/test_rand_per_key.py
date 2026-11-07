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

from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandScaleIntensityd,
    RandScaleIntensityFixedMeand,
    RandShiftIntensityd,
    RandStdShiftIntensityd,
)
from monai.transforms.spatial.dictionary import RandAxisFlipd, RandGridDistortiond, RandRotated, RandZoomd
from tests.test_utils import assert_allclose

KEYS = ["img1", "img2"]

TESTS = [
    (RandGaussianNoised, {}),
    (RandShiftIntensityd, {"offsets": 0.5}),
    (RandStdShiftIntensityd, {"factors": 0.5}),
    (RandScaleIntensityd, {"factors": 0.5}),
    (RandScaleIntensityFixedMeand, {"factors": 0.5}),
    (RandBiasFieldd, {}),
    (RandAdjustContrastd, {}),
    (RandGaussianSmoothd, {}),
    (RandGaussianSharpend, {}),
    (RandHistogramShiftd, {"num_control_points": (5, 20)}),
    (RandGibbsNoised, {}),
    (RandAxisFlipd, {}),
    (RandRotated, {"range_x": 1.0, "range_y": 1.0, "range_z": 1.0}),
    (RandZoomd, {"min_zoom": 0.7, "max_zoom": 1.3}),
    (RandGridDistortiond, {"num_cells": 3, "distort_limit": 0.2}),
]


class TestRandPerKey(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls, kwargs) for cls, kwargs in TESTS])
    def test_shared_default(self, _, cls, kwargs):
        t = cls(keys=KEYS, prob=1.0, **kwargs)
        t.set_random_state(0)
        img = torch.rand(1, 8, 8, 8) + 1.0
        out = t({k: img.clone() for k in KEYS})
        assert_allclose(out["img1"], out["img2"], type_test=False)

    @parameterized.expand([(cls.__name__, cls, kwargs) for cls, kwargs in TESTS])
    def test_independent_per_key(self, _, cls, kwargs):
        # independent draws may coincide for a single seed (e.g. RandAxisFlipd's discrete axis),
        # so only require that some deterministic seed yields divergent per-key outputs
        img = torch.rand(1, 8, 8, 8) + 1.0
        differs = False
        for seed in range(10):
            t = cls(keys=KEYS, prob=1.0, randomize_per_key=True, **kwargs)
            t.set_random_state(seed)
            out = t({k: img.clone() for k in KEYS})
            if not torch.allclose(out["img1"], out["img2"]):
                differs = True
                break
        self.assertTrue(differs)


if __name__ == "__main__":
    unittest.main()
