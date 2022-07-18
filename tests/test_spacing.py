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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.data.meta_obj import set_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import affine_to_spacing
from monai.transforms import Spacing
from monai.utils import ensure_tuple, fall_back_tuple
from tests.utils import TEST_DEVICES, assert_allclose

TESTS = []
for device in TEST_DEVICES:
    TESTS.append(
        [
            {"pixdim": (1.0, 1.5), "padding_mode": "zeros", "dtype": float},
            torch.arange(4).reshape((1, 2, 2)) + 1.0,  # data
            torch.eye(4),
            {},
            torch.tensor([[[1.0, 1.0], [3.0, 2.0]]]),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": 1.0, "padding_mode": "zeros", "dtype": float},
            torch.ones((1, 2, 1, 2)),  # data
            torch.eye(4),
            {},
            torch.tensor([[[[1.0, 1.0]], [[1.0, 1.0]]]]),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (1.0, 1.0, 1.0), "padding_mode": "zeros", "dtype": float},
            torch.ones((1, 2, 1, 2)),  # data
            torch.eye(4),
            {},
            torch.tensor([[[[1.0, 1.0]], [[1.0, 1.0]]]]),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (1.0, 0.2, 1.5), "diagonal": False, "padding_mode": "zeros", "align_corners": True},
            torch.ones((1, 2, 1, 2)),  # data
            torch.tensor([[2, 1, 0, 4], [-1, -3, 0, 5], [0, 0, 2.0, 5], [0, 0, 0, 1]]),
            {},
            torch.tensor([[[[0.95527864, 0.95527864]], [[1.0, 1.0]], [[1.0, 1.0]]]]),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (3.0, 1.0), "padding_mode": "zeros"},
            torch.arange(24).reshape((2, 3, 4)),  # data
            torch.as_tensor(np.diag([-3.0, 0.2, 1.5, 1])),
            {},
            torch.tensor([[[0, 0], [4, 0], [8, 0]], [[12, 0], [16, 0], [20, 0]]]),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (3.0, 1.0), "padding_mode": "zeros"},
            torch.arange(24).reshape((2, 3, 4)),  # data
            torch.eye(4),
            {},
            torch.tensor([[[0, 1, 2, 3], [0, 0, 0, 0]], [[12, 13, 14, 15], [0, 0, 0, 0]]]),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (1.0, 1.0), "align_corners": True},
            torch.arange(24).reshape((2, 3, 4)),  # data
            torch.eye(4),
            {},
            torch.tensor(
                [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
            ),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (4.0, 5.0, 6.0)},
            torch.arange(24).reshape((1, 2, 3, 4)),  # data
            torch.tensor([[-4, 0, 0, 4], [0, 5, 0, -5], [0, 0, 6, -6], [0, 0, 0, 1]]),
            {},
            torch.arange(24).reshape((1, 2, 3, 4)),  # data
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (4.0, 5.0, 6.0), "diagonal": True},
            torch.arange(24).reshape((1, 2, 3, 4)),  # data
            torch.tensor([[-4, 0, 0, 4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]),
            {},
            torch.tensor(
                [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
            ),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (4.0, 5.0, 6.0), "padding_mode": "border", "diagonal": True},
            torch.arange(24).reshape((1, 2, 3, 4)),  # data
            torch.tensor([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]),
            {},
            torch.tensor(
                [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
            ),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (4.0, 5.0, 6.0), "padding_mode": "border", "diagonal": True},
            torch.arange(24).reshape((1, 2, 3, 4)),  # data
            torch.tensor([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]),
            {"mode": "nearest"},
            torch.tensor(
                [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
            ),
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (1.9, 4.0), "padding_mode": "zeros", "diagonal": True},
            torch.arange(24).reshape((1, 4, 6)),  # data
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
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (5.0, 3.0), "padding_mode": "border", "diagonal": True, "dtype": torch.float32},
            torch.arange(24).reshape((1, 4, 6)),  # data
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
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": (5.0, 3.0), "padding_mode": "zeros", "diagonal": True, "dtype": torch.float32},
            torch.arange(24).reshape((1, 4, 6)),  # data
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
            *device,
        ]
    )
    TESTS.append(
        [
            {"pixdim": [-1, -1, 0.5], "padding_mode": "zeros", "dtype": float},
            torch.ones((1, 2, 1, 2)),  # data
            torch.eye(4),
            {},
            torch.tensor([[[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]]]),
            *device,
        ]
    )
    TESTS.append(  # 5D input
        [
            {"pixdim": [-1, -1, 0.5], "padding_mode": "zeros", "dtype": float, "align_corners": True},
            torch.ones((1, 2, 2, 2, 1)),  # data
            torch.eye(4),
            {},
            torch.ones((1, 2, 2, 3, 1)),
            *device,
        ]
    )

TESTS_TORCH = []
for track_meta in (False, True):
    for device in TEST_DEVICES:
        TESTS_TORCH.append([[1.2, 1.3, 0.9], torch.zeros((1, 3, 4, 5)), track_meta, *device])


class TestSpacingCase(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_spacing(self, init_param, img, affine, data_param, expected_output, device):
        img = MetaTensor(img, affine=affine).to(device)
        res: MetaTensor = Spacing(**init_param)(img, **data_param)
        self.assertEqual(img.device, res.device)

        assert_allclose(res, expected_output, atol=1e-1, rtol=1e-1)
        sr = min(len(res.shape) - 1, 3)
        if isinstance(init_param["pixdim"], float):
            init_param["pixdim"] = [init_param["pixdim"]] * sr
        init_pixdim = ensure_tuple(init_param["pixdim"])
        init_pixdim = init_param["pixdim"][:sr]
        norm = affine_to_spacing(res.affine, sr).cpu().numpy()
        assert_allclose(fall_back_tuple(init_pixdim, norm), norm, type_test=False)

    @parameterized.expand(TESTS_TORCH)
    def test_spacing_torch(self, pixdim, img: torch.Tensor, track_meta: bool, device):
        set_track_meta(track_meta)
        tr = Spacing(pixdim=pixdim)
        img = img.to(device)
        res = tr(img)
        if track_meta:
            self.assertIsInstance(res, MetaTensor)
            new_spacing = affine_to_spacing(res.affine, 3)
            assert_allclose(new_spacing, pixdim, type_test=False)
            self.assertNotEqual(img.shape, res.shape)
        else:
            self.assertIsInstance(res, torch.Tensor)
            self.assertNotIsInstance(res, MetaTensor)
            self.assertNotEqual(img.shape, res.shape)

    @parameterized.expand(TEST_DEVICES)
    def test_inverse(self, device):
        img_t = torch.rand((1, 10, 9, 8), dtype=torch.float32, device=device)
        affine = torch.tensor(
            [[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32, device="cpu"
        )
        meta = {"fname": "somewhere"}
        img = MetaTensor(img_t, affine=affine, meta=meta)
        tr = Spacing(pixdim=[1.1, 1.2, 0.9])
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


if __name__ == "__main__":
    unittest.main()
