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
from itertools import product
from unittest.mock import Mock

import numpy as np
from parameterized import parameterized

from monai.data import create_test_image_2d, create_test_image_3d
from monai.utils import set_determinism

TEST_CASES = [
    [2, {"width": 64, "height": 64, "rad_max": 10, "rad_min": 4}, 0.1479004, 0.739502, (64, 64), 5],
    [
        2,
        {"width": 28, "height": 32, "num_objs": 3, "rad_max": 5, "rad_min": 1, "noise_max": 0.2},
        0.1709315,
        0.4040179,
        (32, 28),
        5,
    ],
    [
        3,
        {"width": 64, "height": 64, "depth": 45, "num_seg_classes": 3, "channel_dim": -1, "rad_max": 10, "rad_min": 4},
        0.025132,
        0.0753961,
        (64, 64, 45, 1),
        3,
    ],
]

INSTANCE_ID_CASES = [
    [2, {"width": 64, "height": 64, "num_objs": 5, "rad_max": 10, "rad_min": 4}],
    [3, {"width": 40, "height": 40, "depth": 40, "num_objs": 4, "rad_max": 8, "rad_min": 3, "channel_dim": -1}],
]


class TestDiceCELoss(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_create_test_image(self, dim, input_param, expected_img, expected_seg, expected_shape, expected_max_cls):
        """Verify synthetic image shapes, label classes, and deterministic means.

        Args:
            dim: Spatial dimensionality of the generator.
            input_param: Keyword arguments passed to the generator.
            expected_img: Expected mean image intensity.
            expected_seg: Expected mean segmentation label.
            expected_shape: Expected image shape.
            expected_max_cls: Expected maximum segmentation class.
        """
        set_determinism(seed=0)
        if dim == 2:
            img, seg = create_test_image_2d(**input_param)
        else:  # dim == 3
            img, seg = create_test_image_3d(**input_param)
        self.assertEqual(img.shape, expected_shape)
        self.assertEqual(seg.max(), expected_max_cls)
        np.testing.assert_allclose(img.mean(), expected_img, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(seg.mean(), expected_seg, atol=1e-7, rtol=1e-7)

    @parameterized.expand(INSTANCE_ID_CASES)
    def test_return_instance_id(self, dim, input_param):
        """Verify instance mask shape, dtype, ID bounds, and foreground alignment.

        Args:
            dim: Spatial dimensionality of the generator.
            input_param: Keyword arguments passed to the generator.
        """
        set_determinism(seed=0)
        if dim == 2:
            img, seg, instance_ids = create_test_image_2d(**input_param, return_instance_id=True)
        else:  # dim == 3
            img, seg, instance_ids = create_test_image_3d(**input_param, return_instance_id=True)

        self.assertEqual(img.shape, seg.shape)
        self.assertEqual(instance_ids.shape, seg.shape)
        self.assertEqual(instance_ids.dtype, np.int32)
        unique_ids = np.unique(instance_ids)
        self.assertGreaterEqual(len(unique_ids), 2)
        self.assertEqual(unique_ids[0], 0)
        self.assertTrue(np.all(unique_ids <= input_param["num_objs"]))
        np.testing.assert_array_equal(instance_ids > 0, seg > 0)

    @parameterized.expand(product((2, 3), (0, 2, 12), (None, 0, -1)))
    def test_instance_id_overlap(self, dim, offset, channel_dim):
        """Check distinct IDs and later-object precedence at fixed object positions."""
        generator = create_test_image_2d if dim == 2 else create_test_image_3d
        centers = [(8,) * dim, (8 + offset,) + (8,) * (dim - 1)]
        rs = Mock(spec=np.random.RandomState, wraps=np.random.RandomState(0))
        rs.randint.side_effect = [value for center in centers for value in (*center, 3)]
        image, labels, instance_ids = generator(
            *((32,) * dim),
            num_objs=2,
            rad_min=3,
            rad_max=4,
            num_seg_classes=1,
            channel_dim=channel_dim,
            random_state=rs,
            return_instance_id=True,
        )

        self.assertEqual(image.shape, labels.shape)
        self.assertEqual(instance_ids.shape, labels.shape)
        ids = instance_ids.squeeze()
        self.assertEqual(ids[(0,) * dim], 0)
        self.assertEqual(ids[centers[1]], 2)
        first_only = (5,) + (8,) * (dim - 1)
        self.assertEqual(ids[first_only], 2 if offset == 0 else 1)
        if offset <= 2:
            self.assertEqual(ids[centers[0]], 2)
        expected_ids = [0, 2] if offset == 0 else [0, 1, 2]
        np.testing.assert_array_equal(np.unique(ids), expected_ids)
        np.testing.assert_array_equal(instance_ids > 0, labels > 0)

    def test_ill_radius(self):
        """Verify invalid radius bounds and image sizes raise ValueError."""
        with self.assertRaisesRegex(ValueError, ""):
            img, seg = create_test_image_2d(32, 32, rad_max=20)
        with self.assertRaisesRegex(ValueError, ""):
            img, seg = create_test_image_3d(32, 32, 32, rad_max=10, rad_min=11)
        with self.assertRaisesRegex(ValueError, ""):
            img, seg = create_test_image_2d(32, 32, rad_max=10, rad_min=0)
        with self.assertRaisesRegex(ValueError, "`rad_min` 0 should be no less than 1"):
            img, seg = create_test_image_3d(32, 32, 32, rad_max=10, rad_min=0)


if __name__ == "__main__":
    unittest.main()
