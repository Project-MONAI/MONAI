import unittest

import torch

from monai.transforms.spatial.functional import spacing
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
        "pixdim": (1.0, 1.0, 1.0) if is_3d else (1.0, 1.0),
        "src_pixdim": (1.0, 1.0, 1.0) if is_3d else (1.0, 1.0),
        "diagonal": False,
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


class TestFunctionalSpacing(unittest.TestCase):

    SPACING_CASES = [
        (
            get_arange_img((32, 32)),
            affine_scale_2d(0.5),
            get_metadata(False, {"pixdim": (2.0, 2.0)}),
            get_metadata(False, {"pixdim": (2.0, 2.0), "shape_override": (1, 16, 16)})
        ),
        (
            get_arange_img((32, 32)),
            affine_scale_2d(0.5),
            get_metadata(False, {"src_pixdim": (0.5, 0.5)}),
            get_metadata(False, {"src_pixdim": (0.5, 0.5), "shape_override": (1, 16, 16)})
        ),
        (
            get_arange_img((32, 32)),
            affine_scale_2d(0.25),
            get_metadata(False, {"pixdim": (2.0, 2.0), "src_pixdim": (0.5, 0.5)}),
            get_metadata(False, {"pixdim": (2.0, 2.0), "src_pixdim": (0.5, 0.5),
                                 "shape_override": (1, 8, 8)})
        ),
        (
            get_arange_img((32, 32)),
            affine_scale_2d(2.0),
            get_metadata(False, {"src_pixdim": (2.0, 2.0)}),
            get_metadata(False, {"src_pixdim": (2.0, 2.0), "shape_override": (1, 64, 64)})
        ),
        (
            get_arange_img((32, 32)),
            affine_scale_2d(2.0),
            get_metadata(False, {"pixdim": (0.5, 0.5)}),
            get_metadata(False, {"pixdim": (0.5, 0.5), "shape_override": (1, 64, 64)})
        ),
        (
            get_arange_img((32, 32)),
            affine_scale_2d(4.0),
            get_metadata(False, {"pixdim": (0.5, 0.5), "src_pixdim": (2.0, 2.0)}),
            get_metadata(False, {"pixdim": (0.5, 0.5), "src_pixdim": (2.0, 2.0),
                                 "shape_override": (1, 128, 128)})
        ),
        (
            get_arange_img((32, 32)),
            affine_scale_2d((0.5, 2.0)),
            get_metadata(False, {"pixdim": (2.0, 1.0), "src_pixdim": (1.0, 2.0)}),
            get_metadata(False, {"pixdim": (2.0, 1.0), "src_pixdim": (1.0, 2.0),
                                 "shape_override": (1, 16, 64)})
        ),
        (
            get_arange_img((32, 32)),
            affine_scale_2d((2.0, 0.5)),
            get_metadata(False, {"pixdim": (1.0, 2.0), "src_pixdim": (2.0, 1.0)}),
            get_metadata(False, {"pixdim": (1.0, 2.0), "src_pixdim": (2.0, 1.0),
                                 "shape_override": (1, 64, 16)})
        ),
        (
            get_arange_img((32, 32, 24)),
            affine_scale_3d(0.5),
            get_metadata(True, {"pixdim": (2.0, 2.0, 2.0)}),
            get_metadata(True, {"pixdim": (2.0, 2.0, 2.0),
                                "shape_override": (1, 16, 16, 12)})
        ),
        (
            get_arange_img((32, 32, 24)),
            affine_scale_3d(0.5),
            get_metadata(True, {"src_pixdim": (0.5, 0.5, 0.5)}),
            get_metadata(True, {"src_pixdim": (0.5, 0.5, 0.5),
                                "shape_override": (1, 16, 16, 12)})
        ),
        (
            get_arange_img((32, 32, 24)),
            affine_scale_3d(0.25),
            get_metadata(True, {"pixdim": (2.0, 2.0, 2.0), "src_pixdim": (0.5, 0.5, 0.5)}),
            get_metadata(True, {"pixdim": (2.0, 2.0, 2.0), "src_pixdim": (0.5, 0.5, 0.5),
                                "shape_override": (1, 8, 8, 6)})
        ),
        (
            get_arange_img((32, 32, 24)),
            affine_scale_3d(2.0),
            get_metadata(True, {"src_pixdim": (2.0, 2.0, 2.0)}),
            get_metadata(True, {"src_pixdim": (2.0, 2.0, 2.0),
                                "shape_override": (1, 64, 64, 48)})
        ),
        (
            get_arange_img((32, 32, 24)),
            affine_scale_3d(2.0),
            get_metadata(True, {"pixdim": (0.5, 0.5, 0.5)}),
            get_metadata(True, {"pixdim": (0.5, 0.5, 0.5),
                                "shape_override": (1, 64, 64, 48)})
        ),
        (
            get_arange_img((32, 32, 24)),
            affine_scale_3d(4.0),
            get_metadata(True, {"pixdim": (0.5, 0.5, 0.5), "src_pixdim": (2.0, 2.0, 2.0)}),
            get_metadata(True, {"pixdim": (0.5, 0.5, 0.5), "src_pixdim": (2.0, 2.0, 2.0),
                                "shape_override": (1, 128, 128, 96)})
        ),
        (
            get_arange_img((32, 32, 24)),
            affine_scale_3d((0.5, 2.0, 1/3.0)),
            get_metadata(True, {"pixdim": (2.0, 1.0, 4.5), "src_pixdim": (1.0, 2.0, 1.5)}),
            get_metadata(True, {"pixdim": (2.0, 1.0, 4.5), "src_pixdim": (1.0, 2.0, 1.5),
                                "shape_override": (1, 16, 64, 8)})
        ),
        (
            get_arange_img((32, 32, 24)),
            affine_scale_3d((2.0, 0.5, 3.0)),
            get_metadata(True, {"pixdim": (1.0, 2.0, 1.5), "src_pixdim": (2.0, 1.0, 4.5)}),
            get_metadata(True, {"pixdim": (1.0, 2.0, 1.5), "src_pixdim": (2.0, 1.0, 4.5),
                                "shape_override": (1, 64, 16, 72)})
        ),
    ]

    def _test_functional_spacing_impl(
            self, img, expected_transform, call_params, expected_metadata
    ):
        img_, transform_, metadata = spacing(img, **call_params)
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

    def test_functional_spacing(self):
        for icase, case in enumerate(self.SPACING_CASES):
            with self.subTest(f"{icase}"):
                self._test_functional_spacing_impl(*case)
