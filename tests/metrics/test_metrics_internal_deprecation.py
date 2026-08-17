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

import pathlib
import unittest
import warnings

import torch

import monai
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric

_MONAI_ROOT = str(pathlib.Path(monai.__file__).parent.resolve())


def _internal_deprecation_warnings(fn):
    """Collect deprecation-style warnings raised from inside the MONAI package itself.

    Warnings originating in third-party packages are ignored, so this is not affected by
    unrelated deprecations in torch or numpy.

    Args:
        fn: zero-argument callable to invoke while warnings are being recorded.

    Returns:
        list of `warnings.WarningMessage`: the recorded `DeprecationWarning` and
        `FutureWarning` entries whose originating file lies inside the MONAI package.
        Empty when the call raised no MONAI-internal deprecation warnings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [
        w
        for w in caught
        if issubclass(w.category, (DeprecationWarning, FutureWarning))
        and _MONAI_ROOT in str(pathlib.Path(w.filename).resolve())
    ]


class TestMetricsNoInternalDeprecationWarnings(unittest.TestCase):
    """Metrics must not trigger MONAI's own deprecation warnings via internal call sites.

    `SurfaceDistanceMetric` and `HausdorffDistanceMetric` both route through
    `monai.metrics.utils.get_edge_surface_distance`. If that helper passes a deprecated
    argument to `get_mask_edges`, every metric computation emits a warning that the caller
    never triggered and cannot suppress.
    """

    def test_surface_and_hausdorff_emit_no_internal_deprecation_warnings(self):
        """Assert neither metric raises a MONAI-internal deprecation warning.

        Computes each metric on a fixed pair of 2D binary masks and fails if any
        `DeprecationWarning` or `FutureWarning` originating inside MONAI is recorded.
        """
        pred = torch.zeros(1, 1, 32, 32)
        pred[..., :16, :] = 1
        gt = torch.zeros(1, 1, 32, 32)
        gt[..., :20, :] = 1

        for metric in (SurfaceDistanceMetric(), HausdorffDistanceMetric()):
            with self.subTest(metric=type(metric).__name__):
                found = _internal_deprecation_warnings(lambda m=metric: m(pred, gt))
                self.assertEqual(
                    [str(w.message) for w in found],
                    [],
                    f"{type(metric).__name__} raised MONAI-internal deprecation warning(s)",
                )


if __name__ == "__main__":
    unittest.main()
