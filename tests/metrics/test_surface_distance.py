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

from monai.metrics import SurfaceDistanceMetric
from monai.metrics.utils import get_mask_edges, get_surface_distance
from monai.utils import optional_import

distance_transform_edt, has_scipy = optional_import("scipy.ndimage", name="distance_transform_edt")

_device = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_spherical_seg_3d(
    radius: float = 20.0,
    centre: tuple[int, int, int] = (49, 49, 49),
    im_shape: tuple[int, int, int] = (99, 99, 99),
    im_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Return a 3D image with a sphere inside. Voxel values will be
    1 inside the sphere, and 0 elsewhere.

    Args:
        radius: radius of sphere (in terms of number of voxels, can be partial)
        centre: location of sphere centre.
        im_shape: shape of image to create.
        im_spacing: spacing of image to create.

    See also:
        :py:meth:`~create_test_image_3d`
    """
    # Create image
    image = np.zeros(im_shape, dtype=np.int32)
    spy, spx, spz = np.ogrid[: im_shape[0], : im_shape[1], : im_shape[2]]

    spy = spy.astype(float) * im_spacing[0]
    spx = spx.astype(float) * im_spacing[1]
    spz = spz.astype(float) * im_spacing[2]

    spy -= centre[0]
    spx -= centre[1]
    spz -= centre[2]

    circle = (spx * spx + spy * spy + spz * spz) <= radius * radius

    image[circle] = 1
    image[~circle] = 0
    return image


test_spacing = (0.85, 1.2, 0.9)
TEST_CASES = [
    [[create_spherical_seg_3d(), create_spherical_seg_3d(), "euclidean", None], [0, 0]],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 20, 20)),
            create_spherical_seg_3d(radius=20, centre=(19, 19, 19)),
            "taxicab",
        ],
        [1.0380029806259314, 1.0380029806259314],
    ],
    [
        [
            create_spherical_seg_3d(radius=33, centre=(19, 33, 22)),
            create_spherical_seg_3d(radius=33, centre=(20, 33, 22)),
            "euclidean",
            None,
        ],
        [0.350217, 0.3483278807706289],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            "euclidean",
            None,
        ],
        [15.117741, 12.040033513150455],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            "chessboard",
        ],
        [11.492719, 9.605067064083457],
    ],
    [
        [
            create_spherical_seg_3d(radius=20, centre=(20, 33, 22)),
            create_spherical_seg_3d(radius=40, centre=(20, 33, 22)),
            "taxicab",
        ],
        [20.214613, 12.432687531048186],
    ],
    [
        [np.zeros([99, 99, 99]), create_spherical_seg_3d(radius=40, centre=(20, 33, 22)), "euclidean", None],
        [np.inf, np.inf],
    ],
    [[create_spherical_seg_3d(), np.zeros([99, 99, 99]), "taxicab"], [np.inf, np.inf]],
    [
        [
            create_spherical_seg_3d(radius=33, centre=(42, 45, 52), im_spacing=test_spacing),
            create_spherical_seg_3d(radius=33, centre=(43, 45, 52), im_spacing=test_spacing),
            "euclidean",
            test_spacing,
        ],
        [0.4951, 0.4951],
    ],
]

TEST_CASES_NANS = [
    [
        [
            # both pred and gt do not have foreground, spacing is None, metric and not_nans should be 0
            np.zeros([99, 99, 99]),
            np.zeros([99, 99, 99]),
            None,
        ]
    ],
    [
        [
            # both pred and gt do not have foreground, spacing is not None, metric and not_nans should be 0
            np.zeros([99, 99, 99]),
            np.zeros([99, 99, 99]),
            test_spacing,
        ]
    ],
]


class TestAllSurfaceMetrics(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_value(self, input_data, expected_value):
        if len(input_data) == 3:
            [seg_1, seg_2, metric] = input_data
            spacing = None
        else:
            [seg_1, seg_2, metric, spacing] = input_data

        ct = 0
        seg_1 = torch.tensor(seg_1, device=_device)
        seg_2 = torch.tensor(seg_2, device=_device)
        for symmetric in [True, False]:
            sur_metric = SurfaceDistanceMetric(include_background=False, symmetric=symmetric, distance_metric=metric)
            # shape of seg_1, seg_2 are: HWD, converts to BNHWD
            batch, n_class = 2, 3
            batch_seg_1 = seg_1.unsqueeze(0).unsqueeze(0).repeat([batch, n_class, 1, 1, 1])
            batch_seg_2 = seg_2.unsqueeze(0).unsqueeze(0).repeat([batch, n_class, 1, 1, 1])
            sur_metric(batch_seg_1, batch_seg_2, spacing=spacing)
            result = sur_metric.aggregate()
            expected_value_curr = expected_value[ct]
            np.testing.assert_allclose(expected_value_curr, result.cpu(), rtol=1e-5)
            np.testing.assert_equal(result.device, seg_1.device)
            ct += 1

    @parameterized.expand(TEST_CASES_NANS)
    def test_nans(self, input_data):
        [seg_1, seg_2, spacing] = input_data
        seg_1 = torch.tensor(seg_1)
        seg_2 = torch.tensor(seg_2)
        sur_metric = SurfaceDistanceMetric(include_background=False, get_not_nans=True)
        # test list of channel-first Tensor
        batch_seg_1 = [seg_1.unsqueeze(0)]
        batch_seg_2 = [seg_2.unsqueeze(0)]
        sur_metric(batch_seg_1, batch_seg_2, spacing=spacing)
        result, not_nans = sur_metric.aggregate(reduction="mean")
        np.testing.assert_allclose(0, result, rtol=1e-5)
        np.testing.assert_allclose(0, not_nans, rtol=1e-5)


KDTREE_SPACINGS = [["isotropic_default", None], ["isotropic", (1.0, 1.0, 1.0)], ["anisotropic", (1.0, 2.5, 0.5)]]


def _edge_masks(seed=0):
    # two offset spheres plus a few scattered false positives in the prediction, so the
    # surfaces are non-trivially apart and an outlier expands the cropped bounding box.
    gt = create_spherical_seg_3d(radius=20, centre=(30, 30, 30))
    pred = create_spherical_seg_3d(radius=20, centre=(32, 31, 30))
    rng = np.random.RandomState(seed)
    for _ in range(5):
        pred[tuple(rng.randint(0, s) for s in pred.shape)] = 1
    edges_pred, edges_gt = get_mask_edges(pred, gt)
    return np.asarray(edges_pred, dtype=bool), np.asarray(edges_gt, dtype=bool)


@unittest.skipUnless(has_scipy, "Requires scipy.")
class TestSurfaceDistanceKDTreeMatchesEDT(unittest.TestCase):
    @parameterized.expand(KDTREE_SPACINGS)
    def test_cpu_kdtree_euclidean_distances_match_dense_edt(self, _name, spacing):
        edges_pred, edges_gt = _edge_masks()
        result = np.asarray(get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean", spacing=spacing))
        reference = distance_transform_edt(~edges_gt, sampling=spacing)[edges_pred]
        # same multiset of distances (downstream metrics only use max/percentile/mean)
        np.testing.assert_allclose(np.sort(result), np.sort(reference), rtol=1e-5, atol=1e-5)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, reference.shape)

    def test_torch_input_preserves_type_device_and_matches_dense_edt(self):
        edges_pred, edges_gt = _edge_masks()
        spacing = (1.0, 2.5, 0.5)
        seg_pred, seg_gt = torch.as_tensor(edges_pred), torch.as_tensor(edges_gt)
        result = get_surface_distance(seg_pred, seg_gt, distance_metric="euclidean", spacing=spacing)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device, seg_pred.device)
        reference = distance_transform_edt(~edges_gt, sampling=spacing)[edges_pred]
        np.testing.assert_allclose(np.sort(result.cpu().numpy()), np.sort(reference), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
