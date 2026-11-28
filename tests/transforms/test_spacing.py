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

from monai.config import USE_COMPILED
from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import affine_to_spacing
from monai.transforms import Spacing
from monai.transforms.spatial.functional import _compiled_unsupported
from monai.utils import fall_back_tuple
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.test_utils import TEST_DEVICES, TEST_NDARRAYS_ALL, assert_allclose, dict_product, skip_if_quick

_TEMPLATE_5_COMPILED = torch.tensor([[[[1.0, 1.0]], [[1.0, 1.0]], [[1.0, 1.0]]]])
_TEMPLATE_5_NATIVE = torch.tensor([[[[0.95527864, 0.95527864]], [[1.0, 1.0]], [[1.0, 1.0]]]])


def _template_5_expected_output(device: torch.device) -> torch.Tensor:
    if USE_COMPILED and not _compiled_unsupported(device):
        return _TEMPLATE_5_COMPILED
    return _TEMPLATE_5_NATIVE


all_template_parts = [
    [
        {"pixdim": (1.0, 1.5), "padding_mode": "zeros", "dtype": float},
        torch.arange(4).reshape((1, 2, 2)) + 1.0,
        torch.eye(4),
        {},
        torch.tensor([[[1.0, 1.0], [3.0, 2.0]]]),
    ],
    [
        {"pixdim": 1.0, "padding_mode": "zeros", "dtype": float},
        torch.ones((1, 2, 1, 2)),
        torch.eye(4),
        {},
        torch.tensor([[[[1.0, 1.0]], [[1.0, 1.0]]]]),
    ],
    [
        {"pixdim": 2.0, "padding_mode": "zeros", "dtype": float},
        torch.arange(4).reshape((1, 2, 2)) + 1.0,
        torch.eye(4),
        {},
        torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
    ],
    [
        {"pixdim": (1.0, 1.0, 1.0), "padding_mode": "zeros", "dtype": float},
        torch.ones((1, 2, 1, 2)),
        torch.eye(4),
        {},
        torch.tensor([[[[1.0, 1.0]], [[1.0, 1.0]]]]),
    ],
    [
        {"pixdim": (1.0, 0.2, 1.5), "diagonal": False, "padding_mode": "zeros", "align_corners": True},
        torch.ones((1, 2, 1, 2)),
        torch.tensor([[2, 1, 0, 4], [-1, -3, 0, 5], [0, 0, 2.0, 5], [0, 0, 0, 1]]),
        {},
        _template_5_expected_output,
    ],
    [
        {"pixdim": (3.0, 1.0), "padding_mode": "zeros"},
        torch.arange(24).reshape((2, 3, 4)),
        torch.as_tensor(np.diag([-3.0, 0.2, 1.5, 1])),
        {},
        torch.tensor([[[0, 0], [4, 0], [8, 0]], [[12, 0], [16, 0], [20, 0]]]),
    ],
    [
        {"pixdim": (3.0, 1.0), "padding_mode": "zeros"},
        torch.arange(24).reshape((2, 3, 4)),
        torch.eye(4),
        {},
        torch.tensor([[[0, 1, 2, 3], [0, 0, 0, 0]], [[12, 13, 14, 15], [0, 0, 0, 0]]]),
    ],
    [
        {"pixdim": (1.0, 1.0), "align_corners": True},
        torch.arange(24).reshape((2, 3, 4)),
        torch.eye(4),
        {},
        torch.tensor(
            [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        ),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0)},
        torch.arange(24).reshape((1, 2, 3, 4)),
        torch.tensor([[-4, 0, 0, 4], [0, 5, 0, -5], [0, 0, 6, -6], [0, 0, 0, 1]]),
        {},
        torch.arange(24).reshape((1, 2, 3, 4)),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "diagonal": True},
        torch.arange(24).reshape((1, 2, 3, 4)),
        torch.tensor([[-4, 0, 0, 4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]),
        {},
        torch.tensor(
            [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
        ),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "padding_mode": "border", "diagonal": True},
        torch.arange(24).reshape((1, 2, 3, 4)),
        torch.tensor([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]),
        {},
        torch.tensor(
            [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
        ),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "padding_mode": "border", "diagonal": True},
        torch.arange(24).reshape((1, 2, 3, 4)),
        torch.tensor([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]),
        {"mode": "nearest"},
        torch.tensor(
            [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
        ),
    ],
    [
        {"pixdim": (1.9, 4.0), "padding_mode": "zeros", "diagonal": True},
        torch.arange(24).reshape((1, 4, 6)),
        torch.tensor([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]),
        {"mode": "nearest"},
        torch.tensor(
            [
                [
                    [18.0, 19.0, 20.0, 20.0, 21.0, 22.0, 23.0],
                    [18.0, 19.0, 20.0, 20.0, 21.0, 22.0, 23.0],
                    [12.0, 13.0, 14.0, 14.0, 15.0, 16.0, 17.0],
                    [12.0, 13.0, 14.0, 14.0, 15.0, 16.0, 17.0],
                    [6.0, 7.0, 8.0, 8.0, 9.0, 10.0, 11.0],
                    [6.0, 7.0, 8.0, 8.0, 9.0, 10.0, 11.0],
                    [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0],
                ]
            ]
        ),
    ],
    [
        {"pixdim": (5.0, 3.0), "padding_mode": "border", "diagonal": True, "dtype": torch.float32},
        torch.arange(24).reshape((1, 4, 6)),
        torch.tensor([[-4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]),
        {"mode": "bilinear"},
        torch.tensor(
            [
                [
                    [18.0, 18.6, 19.2, 19.8, 20.400002, 21.0, 21.6, 22.2, 22.8],
                    [10.5, 11.1, 11.700001, 12.299999, 12.900001, 13.5, 14.1, 14.700001, 15.3],
                    [3.0, 3.6000001, 4.2000003, 4.8, 5.4000006, 6.0, 6.6000004, 7.200001, 7.8],
                ]
            ]
        ),
    ],
    [
        {"pixdim": (5.0, 3.0), "padding_mode": "zeros", "diagonal": True, "dtype": torch.float32},
        torch.arange(24).reshape((1, 4, 6)),
        torch.tensor([[-4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]),
        {"mode": "bilinear"},
        torch.tensor(
            [
                [
                    [18.0000, 18.6000, 19.2000, 19.8000, 20.4000, 21.0000, 21.6000, 22.2000, 22.8000],
                    [10.5000, 11.1000, 11.7000, 12.3000, 12.9000, 13.5000, 14.1000, 14.7000, 15.3000],
                    [3.0000, 3.6000, 4.2000, 4.8000, 5.4000, 6.0000, 6.6000, 7.2000, 7.8000],
                ]
            ]
        ),
    ],
    [
        {"pixdim": [-1, -1, 0.5], "padding_mode": "zeros", "dtype": float},
        torch.ones((1, 2, 1, 2)),
        torch.eye(4),
        {},
        torch.tensor([[[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]]]),
    ],
    [  # 5D input
        {"pixdim": [-1, -1, 0.5], "padding_mode": "zeros", "dtype": float, "align_corners": True},
        torch.ones((1, 2, 2, 2, 1)),
        torch.eye(4),
        {},
        torch.ones((1, 2, 2, 3, 1)),
    ],
    [  # 5D input
        {"pixdim": 0.5, "padding_mode": "constant", "mode": "nearest", "scale_extent": True},
        torch.ones((1, 368, 336, 368)),
        torch.tensor(
            [
                [0.41, 0.005, 0.008, -79.7],
                [-0.0049, 0.592, 0.0664, -57.4],
                [-0.0073, -0.0972, 0.404, -32.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        {},
        torch.ones((1, 302, 403, 301)),
    ],
]
TESTS: list[list] = [
    params["template"] + [*params["device_val"]]
    for params in dict_product(template=all_template_parts, device_val=TEST_DEVICES)
]

TESTS_TORCH = [
    [[1.2, 1.3, 0.9], params["p"](torch.zeros((1, 3, 4, 5))), params["track_meta"]]
    for params in dict_product(track_meta=[False, True], p=TEST_NDARRAYS_ALL)
]

TEST_INVERSE = [
    [*params["d"], params["recompute"], params["align"], params["scale_extent"]]
    for params in dict_product(d=TEST_DEVICES, recompute=[False, True], align=[False, True], scale_extent=[False, True])
]

TEST_RECOMPUTE = [[*params["d"], params["align"]] for params in dict_product(d=TEST_DEVICES, align=[False, True])]


@skip_if_quick
class TestSpacingCase(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_spacing(
        self,
        init_param: dict,
        img: torch.Tensor,
        affine: torch.Tensor,
        data_param: dict,
        expected_output: torch.Tensor,
        device: torch.device,
    ):
        img = MetaTensor(img, affine=affine).to(device)
        tr = Spacing(**init_param)
        call_param = data_param.copy()
        call_param["data_array"] = img
        res: MetaTensor = tr(**call_param)  # type: ignore
        self.assertEqual(img.device, res.device)

        test_resampler_lazy(tr, res, init_param=init_param, call_param=call_param)

        if callable(expected_output):
            expected_output = expected_output(device)
        assert_allclose(res, expected_output, atol=1e-1, rtol=1e-1)
        sr = min(len(res.shape) - 1, 3)
        if isinstance(init_param["pixdim"], float):
            init_param["pixdim"] = [init_param["pixdim"]] * sr
        init_pixdim = init_param["pixdim"][:sr]
        norm = affine_to_spacing(res.affine, sr).cpu().numpy()
        assert_allclose(fall_back_tuple(init_pixdim, norm), norm, type_test=False)  # type: ignore

    @parameterized.expand(TESTS_TORCH)
    def test_spacing_torch(self, pixdim, img, track_meta: bool):
        set_track_meta(track_meta)
        init_param = {"pixdim": pixdim}
        tr = Spacing(**init_param)
        call_param = {"data_array": img}
        res = tr(**call_param)

        if track_meta:
            self.assertIsInstance(res, MetaTensor)
            new_spacing = affine_to_spacing(res.affine, 3)  # type: ignore
            assert_allclose(new_spacing, pixdim, type_test=False)
            self.assertNotEqual(img.shape, res.shape)
            test_resampler_lazy(tr, res, init_param=init_param, call_param=call_param)
        else:
            self.assertIsInstance(res, torch.Tensor)
            self.assertNotIsInstance(res, MetaTensor)
            self.assertNotEqual(img.shape, res.shape)
        set_track_meta(True)

    @parameterized.expand(TEST_INVERSE)
    def test_inverse(self, device, recompute, align, scale_extent):
        img_t = torch.rand((1, 10, 9, 8), dtype=torch.float32, device=device)
        affine = torch.tensor(
            [[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cpu"
        )
        meta = {"fname": "somewhere"}
        img = MetaTensor(img_t, affine=affine, meta=meta)
        tr = Spacing(pixdim=[1.1, 1.2, 0.9], recompute_affine=recompute, align_corners=align, scale_extent=scale_extent)
        # check that image and affine have changed
        img = tr(img)
        self.assertNotEqual(img.shape, img_t.shape)
        l2_norm_affine = ((affine - img.affine) ** 2).sum() ** 0.5
        self.assertGreater(l2_norm_affine, 5e-2)
        # check that with inverse, image affine are back to how they were
        img = tr.inverse(img)
        self.assertEqual(img.applied_operations, [])
        self.assertEqual(img.shape, img_t.shape)
        l2_norm_affine = ((affine - img.affine) ** 2).sum() ** 0.5
        self.assertLess(l2_norm_affine, 5e-2)

    @parameterized.expand(TEST_INVERSE)
    def test_inverse_mn_mx(self, device, recompute, align, scale_extent):
        img_t = torch.rand((1, 10, 9, 8), dtype=torch.float32, device=device)
        affine = torch.tensor(
            [[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cpu"
        )
        img = MetaTensor(img_t, affine=affine, meta={"fname": "somewhere"})
        choices = [(None, None), [1.2, None], [None, 0.7], [0.7, 0.9]]
        idx = np.random.choice(range(len(choices)), size=1)[0]
        tr = Spacing(
            pixdim=[1.1, 1.2, 0.9],
            recompute_affine=recompute,
            align_corners=align,
            scale_extent=scale_extent,
            min_pixdim=[0.9, None, choices[idx][0]],
            max_pixdim=[1.1, 1.1, choices[idx][1]],
        )
        img_out = tr(img)
        if isinstance(img_out, MetaTensor):
            if recompute:
                # scale_affine now matches the resampler's align_corners (see Spacing.__call__).
                expected = [1.0, 1.142857, 0.875] if align else [1.0, 1.125, 0.888889]
            else:
                expected = [1.0, 1.2, 0.9]
            assert_allclose(img_out.pixdim, expected, type_test=False, rtol=1e-4)
        img_out = tr.inverse(img_out)
        self.assertEqual(img_out.applied_operations, [])
        self.assertEqual(img_out.shape, img_t.shape)
        self.assertLess(((affine - img_out.affine) ** 2).sum() ** 0.5, 5e-2)

    @parameterized.expand(TEST_RECOMPUTE)
    def test_recompute_affine_keeps_geometry(self, device, align):
        # non-unit spacing, a flipped axis and a non-zero origin
        affine = torch.tensor(
            [[-1.2, 0, 0, 90], [0, 0.9, 0, -120], [0, 0, 1.5, 30], [0, 0, 0, 1]], dtype=torch.float64, device="cpu"
        )
        in_shape = (32, 36, 28)
        # smooth, asymmetric image so that a misplaced or flipped grid shows up in the voxel values
        grid = torch.meshgrid(*[torch.linspace(0, 1, s) for s in in_shape], indexing="ij")
        img_t = (2 * grid[0] + torch.cos(2 * grid[1]) + grid[2] ** 2)[None].to(device)
        img = MetaTensor(img_t, affine=affine, meta={"fname": "somewhere"})

        expected = Spacing(pixdim=1.0, align_corners=align)(img)
        tr = Spacing(pixdim=1.0, recompute_affine=True, align_corners=align)
        res = tr(img)
        out_shape = np.asarray(res.shape[1:])
        self.assertEqual(tuple(out_shape), tuple(expected.shape[1:]))

        # pixdim follows from the input spacing and the actual shape change
        in_pixdim = np.array([1.2, 0.9, 1.5])
        if align:
            expected_pixdim = in_pixdim * (np.asarray(in_shape) - 1) / (out_shape - 1)
        else:
            expected_pixdim = in_pixdim * np.asarray(in_shape) / out_shape
        assert_allclose(res.pixdim, expected_pixdim, type_test=False, rtol=1e-4)
        # orientation and origin stay close to the analytic output affine (only quantization differs)
        assert_allclose(res.affine, expected.affine, type_test=False, atol=0.3)
        assert_allclose(np.sign(np.diag(res.affine.cpu().numpy())), [-1, 1, 1, 1], type_test=False)

        # inverse brings the voxel values back onto the input grid
        inv = tr.inverse(res)
        self.assertEqual(inv.shape, img_t.shape)
        assert_allclose(inv.affine, affine, type_test=False, atol=1e-4)
        assert_allclose(inv[0, 3:-3, 3:-3, 3:-3], img_t[0, 3:-3, 3:-3, 3:-3], type_test=False, atol=0.1)

    @parameterized.expand(
        [
            ("flip_x_2", [[-1.0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], (2, 3, 4)),
            ("flip_x_5", [[-1.0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], (5, 3, 4)),
            ("permute_flip", [[0, -1.0, 0, 5], [1.0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], (5, 6, 4)),
        ]
    )
    def test_recompute_affine_diagonal(self, _, affine, shape):
        # with diagonal=True the output grid follows the world axes, so flipped or permuted input axes are
        # reoriented in the data and the recomputed affine has to describe that grid
        affine = torch.tensor(affine, dtype=torch.float64)
        img = MetaTensor(torch.arange(np.prod(shape), dtype=torch.float32).reshape(1, *shape), affine=affine)
        pixdim = np.linalg.norm(affine[:3, :3].numpy(), axis=0)
        expected = Spacing(pixdim=pixdim, diagonal=True)(img)
        res = Spacing(pixdim=pixdim, diagonal=True, recompute_affine=True)(img)
        # no spacing change, so the recomputed affine is the analytic one
        assert_allclose(res, expected, type_test=False)
        assert_allclose(res.affine, expected.affine, type_test=False, atol=1e-6)

        # with a spacing change only the quantization differs from the analytic affine
        expected = Spacing(pixdim=0.4, diagonal=True)(img)
        res = Spacing(pixdim=0.4, diagonal=True, recompute_affine=True)(img)
        assert_allclose(np.sign(res.affine[:3, :3].cpu().numpy()), np.sign(expected.affine[:3, :3]), type_test=False)
        assert_allclose(res.affine, expected.affine, type_test=False, atol=0.3)

    def test_recompute_affine_diagonal_sheared(self):
        # a small pixdim shrinks the off-axis terms of the voxel mapping; a sheared input must still
        # get the orthogonal destination grid
        affine = torch.tensor([[1.0, 0.04, 0], [0, 1.0, 0], [0, 0, 1]], dtype=torch.float64)
        img = MetaTensor(torch.rand(1, 2, 2), affine=affine)
        expected = Spacing(pixdim=0.02, diagonal=True)(img)
        res = Spacing(pixdim=0.02, diagonal=True, recompute_affine=True)(img)
        assert_allclose(res.affine, expected.affine, type_test=False, atol=1e-6)

    def test_property_no_change(self):
        affine = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cpu"
        )
        affine[:3] *= 1 - 1e-4  # make sure it's not exactly target but close to the target
        img = MetaTensor(torch.rand((1, 10, 9, 8), dtype=torch.float32), affine=affine, meta={"fname": "somewhere"})
        tr = Spacing(pixdim=[1.0, 1.0, 1.0])
        tr(img)
        assert_allclose(tr.pixdim, [1.0, 1.0, 1.0], type_test=False)


if __name__ == "__main__":
    unittest.main()
