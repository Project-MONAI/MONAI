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

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import CenterlineDiceMetric, compute_centerline_dice
from tests.test_utils import SkipIfNoModule


def _line_2d(row: int, cols: tuple[int, int], size: int = 7) -> np.ndarray:
    """One-pixel-wide horizontal line; its skeleton is the line itself."""
    img = np.zeros((size, size), dtype=np.float32)
    img[row, cols[0] : cols[1]] = 1.0
    return img


def _to_bc(img: np.ndarray) -> torch.Tensor:
    return torch.tensor(img[None, None])


# hand-computed cases with one-pixel-wide lines (skeleton == line)
REF_LINE = _line_2d(3, (1, 6))  # 5 skeleton pixels
IDENTICAL = [_to_bc(REF_LINE), _to_bc(REF_LINE), {"metric_name": "cldsc"}, [[1.0]]]
IDENTICAL_TPRECS = [_to_bc(REF_LINE), _to_bc(REF_LINE), {"metric_name": "tprec"}, [[1.0]]]
IDENTICAL_TSENS = [_to_bc(REF_LINE), _to_bc(REF_LINE), {"metric_name": "tsens"}, [[1.0]]]
# prediction is a 3-pixel subset of the 5-pixel reference line:
# T_prec = 3/3 = 1, T_sens = 3/5 = 0.6, clDSC = 2*1*0.6/1.6 = 0.75
SUBSET = [_to_bc(_line_2d(3, (2, 5))), _to_bc(REF_LINE), {"metric_name": "cldsc"}, [[0.75]]]
SUBSET_TPRECS = [_to_bc(_line_2d(3, (2, 5))), _to_bc(REF_LINE), {"metric_name": "topology_precision"}, [[1.0]]]
SUBSET_TSENS = [_to_bc(_line_2d(3, (2, 5))), _to_bc(REF_LINE), {"metric_name": "topology_sensitivity"}, [[0.6]]]
# prediction is a superset of the reference: T_prec = 3/5 = 0.6, T_sens = 1
SUPERSET = [_to_bc(REF_LINE), _to_bc(_line_2d(3, (2, 5))), {"metric_name": "cldsc"}, [[0.75]]]
# disjoint lines: no overlap at all
DISJOINT = [_to_bc(_line_2d(1, (1, 6))), _to_bc(REF_LINE), {"metric_name": "cldsc"}, [[0.0]]]
# class absent from both: nan; present in only one: 0
EMPTY_EMPTY = [_to_bc(np.zeros((7, 7), dtype=np.float32)), _to_bc(np.zeros((7, 7), dtype=np.float32)), {}, [[np.nan]]]
EMPTY_PRED = [_to_bc(np.zeros((7, 7), dtype=np.float32)), _to_bc(REF_LINE), {}, [[0.0]]]
EMPTY_REF = [_to_bc(REF_LINE), _to_bc(np.zeros((7, 7), dtype=np.float32)), {}, [[0.0]]]
# alias
ALIAS = [_to_bc(REF_LINE), _to_bc(REF_LINE), {"metric_name": "centerline_dice"}, [[1.0]]]


@SkipIfNoModule("skimage.morphology")
class TestComputeCenterlineDice(unittest.TestCase):
    @parameterized.expand(
        [
            IDENTICAL,
            IDENTICAL_TPRECS,
            IDENTICAL_TSENS,
            SUBSET,
            SUBSET_TPRECS,
            SUBSET_TSENS,
            SUPERSET,
            DISJOINT,
            EMPTY_EMPTY,
            EMPTY_PRED,
            EMPTY_REF,
            ALIAS,
        ]
    )
    def test_value_2d(self, y_pred, y, params, expected):
        result = compute_centerline_dice(y_pred=y_pred, y=y, **params)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-5)

    def test_value_3d(self):
        # one-voxel-wide line along the last axis; Lee skeletonization keeps it intact
        line = np.zeros((7, 7, 7), dtype=np.float32)
        line[3, 3, 1:6] = 1.0
        ref = _to_bc(line)
        pred = _to_bc(line.copy())
        result = compute_centerline_dice(y_pred=pred, y=ref)
        np.testing.assert_allclose(result.cpu().numpy(), [[1.0]], rtol=1e-5)
        # 3-voxel subset: T_prec = 1, T_sens = 3/5
        sub = np.zeros((7, 7, 7), dtype=np.float32)
        sub[3, 3, 2:5] = 1.0
        result = compute_centerline_dice(y_pred=_to_bc(sub), y=ref, metric_name="cldsc")
        np.testing.assert_allclose(result.cpu().numpy(), [[0.75]], rtol=1e-5)

    def test_solid_blob_fallback(self):
        # skimage's Lee method returns an empty skeleton for small solid 3D blobs such as a 2x2x2
        # cube; the mask fallback must still give identical masks a score of 1 (all metrics) and
        # disjoint solid blobs a clDSC of 0
        cube = np.zeros((7, 7, 7), dtype=np.float32)
        cube[2:4, 2:4, 2:4] = 1.0
        for params in ({"metric_name": "cldsc"}, {"metric_name": "tprec"}, {"metric_name": "tsens"}):
            result = compute_centerline_dice(y_pred=_to_bc(cube), y=_to_bc(cube), **params)
            np.testing.assert_allclose(result.cpu().numpy(), [[1.0]], rtol=1e-5)
        shifted = np.zeros((7, 7, 7), dtype=np.float32)
        shifted[4:6, 4:6, 4:6] = 1.0
        result = compute_centerline_dice(y_pred=_to_bc(cube), y=_to_bc(shifted))
        np.testing.assert_allclose(result.cpu().numpy(), [[0.0]], rtol=1e-5)

    def test_thick_structure(self):
        # identical thick masks score 1 regardless of the skeleton shape
        thick = np.zeros((9, 9), dtype=np.float32)
        thick[3:6, 2:7] = 1.0
        result = compute_centerline_dice(y_pred=_to_bc(thick), y=_to_bc(thick), metric_name="cldsc")
        np.testing.assert_allclose(result.cpu().numpy(), [[1.0]], rtol=1e-5)
        # 3x5 bar vs an internal 5-pixel line. The Zhang-Suen skeleton of the bar is the 4-pixel
        # shape {(3,5), (4,2), (4,3), (4,4)}, of which 3 pixels lie on the prediction line:
        # T_prec = 5/5 = 1, T_sens = 3/4 = 0.75, clDSC = 2*1*0.75/1.75 = 0.8571...
        thin = np.zeros((9, 9), dtype=np.float32)
        thin[4, 2:7] = 1.0
        result = compute_centerline_dice(y_pred=_to_bc(thin), y=_to_bc(thick), metric_name="cldsc")
        np.testing.assert_allclose(result.cpu().numpy(), [[6.0 / 7.0]], rtol=1e-5)

    def test_multi_class_include_background(self):
        # channel 0 empty in both (nan), channel 1 identical lines (1.0)
        y_pred = torch.zeros(1, 2, 7, 7)
        y = torch.zeros(1, 2, 7, 7)
        y_pred[0, 1] = torch.tensor(REF_LINE)
        y[0, 1] = torch.tensor(REF_LINE)
        result = compute_centerline_dice(y_pred=y_pred, y=y, include_background=True)
        self.assertTrue(np.isnan(result[0, 0].item()))
        np.testing.assert_allclose(result[0, 1].item(), 1.0, rtol=1e-5)
        result = compute_centerline_dice(y_pred=y_pred, y=y, include_background=False)
        self.assertEqual(result.shape, (1, 1))
        np.testing.assert_allclose(result[0, 0].item(), 1.0, rtol=1e-5)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_centerline_dice(y_pred=_to_bc(REF_LINE), y=_to_bc(REF_LINE), metric_name="unknown")
        with self.assertRaises(ValueError):
            compute_centerline_dice(y_pred=np.zeros((1, 1, 7, 7)), y=_to_bc(REF_LINE))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            compute_centerline_dice(y_pred=torch.zeros(1, 1, 7), y=torch.zeros(1, 1, 7))
        with self.assertRaises(ValueError):
            compute_centerline_dice(y_pred=_to_bc(REF_LINE), y=_to_bc(np.zeros((9, 9), dtype=np.float32)))
        with self.assertRaises(ValueError):
            CenterlineDiceMetric(metric_name="unknown")

    def test_metric_class(self):
        metric = CenterlineDiceMetric(metric_name="cldsc", include_background=True)
        metric(y_pred=_to_bc(REF_LINE), y=_to_bc(REF_LINE))
        metric(y_pred=_to_bc(_line_2d(3, (2, 5))), y=_to_bc(REF_LINE))
        result = metric.aggregate(reduction="mean")
        np.testing.assert_allclose(result.item(), (1.0 + 0.75) / 2, rtol=1e-5)
        # reduction "none" returns the accumulated [B, C] buffer
        metric.reset()
        metric(y_pred=_to_bc(REF_LINE), y=_to_bc(REF_LINE))
        result = metric.aggregate(reduction="none")
        np.testing.assert_allclose(result.cpu().numpy(), [[1.0]], rtol=1e-5)

    def test_metric_class_not_nans(self):
        metric = CenterlineDiceMetric(include_background=True, get_not_nans=True)
        metric(y_pred=_to_bc(np.zeros((7, 7), dtype=np.float32)), y=_to_bc(np.zeros((7, 7), dtype=np.float32)))
        metric(y_pred=_to_bc(REF_LINE), y=_to_bc(REF_LINE))
        result, not_nans = metric.aggregate(reduction="mean")
        np.testing.assert_allclose(result.item(), 1.0, rtol=1e-5)
        self.assertEqual(not_nans.item(), 1)


if __name__ == "__main__":
    unittest.main()
