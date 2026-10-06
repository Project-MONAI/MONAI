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
import warnings

import torch

from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.metrics.utils import get_mask_edges
from monai.utils import optional_import

_, has_scipy = optional_import("scipy.ndimage", name="binary_erosion")

_DEPRECATED_ARG_MESSAGE = ".*always_return_as_numpy.*"


@unittest.skipUnless(has_scipy, "Requires scipy.")
class TestMetricsNoInternalDeprecationWarnings(unittest.TestCase):
    """`SurfaceDistanceMetric` and `HausdorffDistanceMetric` must not trigger deprecation warnings of their own.

    Both metrics route through `get_edge_surface_distance`, which calls `get_mask_edges`. If that helper passes the
    deprecated `always_return_as_numpy` argument, every metric computation warns about an argument the caller never
    passed and cannot suppress. Both metrics compute mask edges via `scipy.ndimage`, so the tests need scipy.
    """

    def setUp(self):
        self.pred = torch.zeros(1, 1, 32, 32)
        self.pred[..., :16, :] = 1
        self.gt = torch.zeros(1, 1, 32, 32)
        self.gt[..., :20, :] = 1

    def test_deprecated_arg_warns_when_passed_explicitly(self):
        """Control: passing the deprecated argument directly does emit a `FutureWarning`.

        Ensures the no-warning checks below are meaningful and not passing vacuously.
        """
        with self.assertWarnsRegex(FutureWarning, "always_return_as_numpy"):
            get_mask_edges(self.pred, self.gt, always_return_as_numpy=False)

    def test_metrics_emit_no_deprecated_arg_warning(self):
        """Neither metric emits the `always_return_as_numpy` `FutureWarning`.

        The warning is escalated to an error, so any internal use of the deprecated argument fails the test.
        """
        for metric in (SurfaceDistanceMetric(), HausdorffDistanceMetric()):
            with self.subTest(metric=type(metric).__name__):
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", message=_DEPRECATED_ARG_MESSAGE, category=FutureWarning)
                    metric(self.pred, self.gt)


if __name__ == "__main__":
    unittest.main()
