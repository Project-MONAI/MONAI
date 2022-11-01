import unittest

import torch

from monai.transforms.spatial.functional import resize
from tests.utils import get_arange_img


def affine_scale_2d(scale):
    scale0, scale1 = scale if isinstance(scale, (tuple, list)) else (scale, scale)
    t = torch.eye(3)
    t[:, 0] = torch.FloatTensor([scale0, 0.0, 0.0])
    t[:, 1] = torch.FloatTensor([0.0, scale1, 0.0])
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
        "size_mode": "all",
        "mode": "nearest",
        "align_corners": False,
        "anti_aliasing": False,
        "anti_aliasing_sigma": None,
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


class TestFunctionalSpacing(unittest.TestCase):

    SPACING_CASES = [
        # 2d - "all"
        (
            get_arange_img((32, 32)),
            affine_scale_2d(0.5),
            get_metadata(False, {"spatial_size": (16, 16)}),
            get_metadata(False, {"spatial_size": (16, 16), "shape_override": (1, 16, 16)})
        ),
        (
            get_arange_img((32, 32)),
            affine_scale_2d(2.0),
            get_metadata(False, {"spatial_size": (64, 64)}),
            get_metadata(False, {"spatial_size": (64, 64), "shape_override": (1, 64, 64)})
        ),
        (
            get_arange_img((32, 16)),
            affine_scale_2d((0.5, 1.0)),
            get_metadata(False, {"spatial_size": (16, 16)}),
            get_metadata(False, {"spatial_size": (16, 16), "shape_override": (1, 16, 16)})
        ),
        # 2d - "longest"
        (
            get_arange_img((32, 16)),
            affine_scale_2d(0.5),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest",
                                 "shape_override": (1, 16, 8)})
        ),
        (
            get_arange_img((16, 32)),
            affine_scale_2d(0.5),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest",
                                 "shape_override": (1, 8, 16)})
        ),
        (
            get_arange_img((32, 16)),
            affine_scale_2d(2.0),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest",
                                 "shape_override": (1, 64, 32)})
        ),
        (
            get_arange_img((16, 32)),
            affine_scale_2d(2.0),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest",
                                 "shape_override": (1, 32, 64)})
        ),
        # 3d - "all"
        (
            get_arange_img((32, 32, 16)),
            affine_scale_3d(0.5),
            get_metadata(False, {"spatial_size": (16, 16, 8)}),
            get_metadata(False, {"spatial_size": (16, 16, 8),
                                 "shape_override": (1, 16, 16, 8)})
        ),
        (
            get_arange_img((32, 32, 16)),
            affine_scale_3d(2.0),
            get_metadata(False, {"spatial_size": (64, 64, 32)}),
            get_metadata(False, {"spatial_size": (64, 64, 32),
                                 "shape_override": (1, 64, 64, 32)})
        ),
        # 3d - "longest"
        (
            get_arange_img((32, 16, 8)),
            affine_scale_3d(0.5),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest",
                                 "shape_override": (1, 16, 8, 4)})
        ),
        (
            get_arange_img((16, 32, 8)),
            affine_scale_3d(0.5),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest",
                                 "shape_override": (1, 8, 16, 4)})
        ),
        (
            get_arange_img((8, 16, 32)),
            affine_scale_3d(0.5),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 16, "size_mode": "longest",
                                 "shape_override": (1, 4, 8, 16)})
        ),
        (
            get_arange_img((32, 16, 8)),
            affine_scale_3d(2.0),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest",
                                 "shape_override": (1, 64, 32, 16)})
        ),
        (
            get_arange_img((16, 32, 8)),
            affine_scale_3d(2.0),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest",
                                 "shape_override": (1, 32, 64, 16)})
        ),
        (
            get_arange_img((8, 16, 32)),
            affine_scale_3d(2.0),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest"}),
            get_metadata(False, {"spatial_size": 64, "size_mode": "longest",
                                 "shape_override": (1, 16, 32, 64)})
        ),
    ]

    def _test_functional_resize_impl(
            self, img, expected_transform, call_params, expected_metadata
    ):
        img_, transform_, metadata = resize(img, **call_params)
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

    def test_functional_resize(self):
        for icase, case in enumerate(self.SPACING_CASES):
            with self.subTest(f"{icase}"):
                self._test_functional_resize_impl(*case)
