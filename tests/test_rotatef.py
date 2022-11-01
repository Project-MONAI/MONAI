import unittest

import math

import torch

from monai.transforms.spatial.functional import rotate
from tests.utils import get_arange_img


def affine_rotate_2d(radians, scale=None):
    t = torch.eye(3)
    t[:, 0] = torch.FloatTensor([math.cos(radians), math.sin(radians), 0.0])
    t[:, 1] = torch.FloatTensor([-math.sin(radians), math.cos(radians), 0.0])
    if scale is not None:
        t[0, 0] *= scale
        t[0, 1] *= scale
        t[1, 0] *= scale
        t[1, 1] *= scale
    return t


def affine_scale_3d(scale):
    scale0, scale1, scale2 = scale if isinstance(scale, (tuple, list)) else (scale, scale, scale)
    t = torch.eye(4)
    t[:, 0] = torch.FloatTensor([scale0, 0.0, 0.0, 0.0])
    t[:, 1] = torch.FloatTensor([0.0, scale1, 0.0, 0.0])
    t[:, 2] = torch.FloatTensor([0.0, 0.0, scale2, 0.0])
    return t


def get_metadata(is_3d=True, overrides=None, remove=None):
    metadata = {
        "angle": 0.0,
        "keep_size": True,
        "mode": "nearest",
        "padding_mode": "zeros",
        "align_corners": False,
        "dtype": torch.float32,
        # "im_extents": None,
        # "shape_override": torch.IntTensor([1, 32, 32]) # shape override shouldn't always be in here
    }
    if overrides is not None:
        for k, v in overrides.items():
            metadata[k] = v
    if remove is not None:
        for k in remove:
            if k in metadata:
                del metadata[k]
    return metadata


class TestFunctionalRotate(unittest.TestCase):

    ROTATE_CASES = [
        # keep_size = True
        (
            get_arange_img((32, 32)),
            affine_rotate_2d(torch.pi / 4),
            get_metadata(False, {"angle": torch.pi / 4}),
            get_metadata(False, {"angle": torch.pi / 4, "shape_override": (1, 32, 32)})
        ),
        (
            get_arange_img((32, 32)),
            affine_rotate_2d(torch.pi / 4, 32/33),
            get_metadata(False, {"angle": torch.pi / 4, "align_corners": True}),
            get_metadata(False, {"angle": torch.pi / 4, "align_corners": True,
                                 "shape_override": (1, 32, 32)})
        ),
        # keep_size = False
        (
            get_arange_img((32, 32)),
            affine_rotate_2d(torch.pi / 4),
            get_metadata(False, {"angle": torch.pi / 4, "keep_size": False}),
            get_metadata(False, {"angle": torch.pi / 4, "keep_size": False,
                                 "shape_override": (1, 45, 45)})
        ),
        (
            get_arange_img((32, 32)),
            affine_rotate_2d(torch.pi / 4, 45/46),
            get_metadata(False, {"angle": torch.pi / 4, "keep_size": False, "align_corners": True}),
            get_metadata(False, {"angle": torch.pi / 4, "keep_size": False, "align_corners": True,
                                 "shape_override": (1, 45, 45)})
        ),
    ]

    def _test_functional_rotate_impl(
            self, img, expected_transform, call_params, expected_metadata
    ):
        img_, transform_, metadata = rotate(img, **call_params)
        self.assertTrue(torch.allclose(transform_, expected_transform),
                        msg=f"{transform_} != {expected_transform}")
        actual_keys = set(metadata.keys())
        expected_keys = set(expected_metadata.keys())
        self.assertSetEqual(actual_keys, expected_keys)
        for k in actual_keys:
            if isinstance(metadata[k], torch.Tensor):
                self.assertTrue(torch.allclose(metadata[k], expected_metadata[k]),
                                msg=f"{metadata[k]} != {expected_metadata[k]} for key {k}")
            else:
                self.assertEqual(metadata[k], expected_metadata[k],
                                 msg=f"{metadata[k]} != {expected_metadata[k]} for key {k}")

    def test_functional_rotate(self):
        for icase, case in enumerate(self.ROTATE_CASES):
            with self.subTest(f"{icase}"):
                self._test_functional_rotate_impl(*case)
