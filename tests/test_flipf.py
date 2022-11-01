import unittest

from typing import Sequence, Union

import torch

from monai.transforms.spatial.functional import flip
from tests.utils import get_arange_img


def affine_flip(dims, axis: Union[int, Sequence[int]]):
    if axis is None:
        return torch.eye(dims + 1)

    if not isinstance(axis, (list, tuple)):
        axis = (axis,)
    t = torch.eye(dims + 1)
    for i in axis:
        t[i, i] = -1
    return t


def get_metadata(overrides=None, remove=None):
    metadata = {
        "spatial_axis": None,
        "shape_override": None,
    }
    if overrides is not None:
        for k, v in overrides.items():
            metadata[k] = v
    if remove is not None:
        for k in remove:
            if k in metadata:
                del metadata[k]
    return metadata


class TestFunctionalSpacing(unittest.TestCase):

    FLIP_CASES = [
        # 2d cases
        (
            get_arange_img((32, 32)), affine_flip(2, (0, 1)),
            get_metadata({"spatial_axis": None}), get_metadata({"spatial_axis": (0, 1)})
        ),
        (
            get_arange_img((32, 32)), affine_flip(2, 0),
            get_metadata({"spatial_axis": 0}), get_metadata({"spatial_axis": 0})
        ),
        (
            get_arange_img((32, 32)), affine_flip(2, 1),
            get_metadata({"spatial_axis": 1}), get_metadata({"spatial_axis": 1})
        ),
        (
            get_arange_img((32, 32)), affine_flip(2, 0),
            get_metadata({"spatial_axis": (0,)}), get_metadata({"spatial_axis": (0,)})
        ),
        (
            get_arange_img((32, 32)), affine_flip(2, 1),
            get_metadata({"spatial_axis": (1,)}), get_metadata({"spatial_axis": (1,)})
        ),
        (
            get_arange_img((32, 32)), affine_flip(2, (0, 1)),
            get_metadata({"spatial_axis": (0, 1)}), get_metadata({"spatial_axis": (0, 1)})
        ),
        (
            get_arange_img((32, 32)), affine_flip(2, (0, 1)),
            get_metadata({"spatial_axis": (1, 0)}), get_metadata({"spatial_axis": (1, 0)})
        ),
        # 3d cases
        (
            get_arange_img((32, 32, 16)), affine_flip(3, (0, 1, 2)),
            get_metadata({"spatial_axis": None}), get_metadata({"spatial_axis": (0, 1, 2)})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, 0),
            get_metadata({"spatial_axis": 0}), get_metadata({"spatial_axis": 0})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, 1),
            get_metadata({"spatial_axis": 1}), get_metadata({"spatial_axis": 1})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, 2),
            get_metadata({"spatial_axis": 2}), get_metadata({"spatial_axis": 2})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, 0),
            get_metadata({"spatial_axis": (0,)}), get_metadata({"spatial_axis": (0,)})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, 1),
            get_metadata({"spatial_axis": (1,)}), get_metadata({"spatial_axis": (1,)})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, 2),
            get_metadata({"spatial_axis": (2,)}), get_metadata({"spatial_axis": (2,)})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, (0, 1)),
            get_metadata({"spatial_axis": (0, 1)}), get_metadata({"spatial_axis": (0, 1)})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, (0, 2)),
            get_metadata({"spatial_axis": (0, 2)}), get_metadata({"spatial_axis": (0, 2)})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, (1, 2)),
            get_metadata({"spatial_axis": (1, 2)}), get_metadata({"spatial_axis": (1, 2)})
        ),
        (
            get_arange_img((32, 32, 16)), affine_flip(3, (0, 1, 2)),
            get_metadata({"spatial_axis": (0, 1, 2)}), get_metadata({"spatial_axis": (0, 1, 2)})
        ),
    ]

    def _test_functional_flip_impl(
            self, img, expected_transform, call_params, expected_metadata
    ):
        img_, transform_, metadata = flip(img, **call_params)
        self.assertTrue(torch.allclose(transform_, expected_transform),
                        msg=f"{transform_} != {expected_transform}")
        actual_keys = set(metadata.keys())
        expected_keys = set(expected_metadata.keys())
        self.assertSetEqual(actual_keys, expected_keys)
        for k in actual_keys:
            if isinstance(metadata[k], torch.Tensor) and metadata[k] is not None:
                self.assertTrue(torch.allclose(metadata[k], expected_metadata[k]),
                                msg=f"{metadata[k]} != {expected_metadata[k]} for key {k}")
            else:
                self.assertEqual(metadata[k], expected_metadata[k],
                                 msg=f"{metadata[k]} != {expected_metadata[k]} for key {k}")

    def test_functional_flip(self):
        for icase, case in enumerate(self.FLIP_CASES):
            with self.subTest(f"{icase}"):
                self._test_functional_flip_impl(*case)