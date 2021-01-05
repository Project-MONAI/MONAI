# Copyright 2020 - 2021 MONAI Consortium
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
from parameterized import parameterized

from monai.transforms import Spacing
from monai.utils import ensure_tuple

TEST_CASES = [
    [
        {"pixdim": (1.0, 1.5, 1.0), "padding_mode": "zeros", "dtype": np.float},
        np.arange(4).reshape((1, 2, 2)) + 1.0,  # data
        {"affine": np.eye(4)},
        np.array([[[1.0, 1.0], [3.0, 2.0]]]),
    ],
    [
        {"pixdim": 1.0, "padding_mode": "zeros", "dtype": np.float},
        np.ones((1, 2, 1, 2)),  # data
        {"affine": np.eye(4)},
        np.array([[[[1.0, 1.0]], [[1.0, 1.0]]]]),
    ],
    [
        {"pixdim": (1.0, 1.0, 1.0), "padding_mode": "zeros", "dtype": np.float},
        np.ones((1, 2, 1, 2)),  # data
        {"affine": np.eye(4)},
        np.array([[[[1.0, 1.0]], [[1.0, 1.0]]]]),
    ],
    [
        {"pixdim": (1.0, 0.2, 1.5), "diagonal": False, "padding_mode": "zeros", "align_corners": True},
        np.ones((1, 2, 1, 2)),  # data
        {"affine": np.array([[2, 1, 0, 4], [-1, -3, 0, 5], [0, 0, 2.0, 5], [0, 0, 0, 1]])},
        np.array([[[[0.95527864, 0.95527864]], [[1.0, 1.0]], [[1.0, 1.0]]]]),
    ],
    [
        {"pixdim": (3.0, 1.0), "padding_mode": "zeros"},
        np.arange(24).reshape((2, 3, 4)),  # data
        {"affine": np.diag([-3.0, 0.2, 1.5, 1])},
        np.array([[[0, 0], [4, 0], [8, 0]], [[12, 0], [16, 0], [20, 0]]]),
    ],
    [
        {"pixdim": (3.0, 1.0), "padding_mode": "zeros"},
        np.arange(24).reshape((2, 3, 4)),  # data
        {},
        np.array([[[0, 1, 2, 3], [0, 0, 0, 0]], [[12, 13, 14, 15], [0, 0, 0, 0]]]),
    ],
    [
        {"pixdim": (1.0, 1.0)},
        np.arange(24).reshape((2, 3, 4)),  # data
        {},
        np.array(
            [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        ),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0)},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {"affine": np.array([[-4, 0, 0, 4], [0, 5, 0, -5], [0, 0, 6, -6], [0, 0, 0, 1]])},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "diagonal": True},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {"affine": np.array([[-4, 0, 0, 4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]])},
        np.array(
            [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
        ),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "padding_mode": "border", "diagonal": True},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {"affine": np.array([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]])},
        np.array(
            [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
        ),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "padding_mode": "border", "diagonal": True},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {"affine": np.array([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), "mode": "nearest"},
        np.array(
            [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
        ),
    ],
    [
        {"pixdim": (1.9, 4.0, 5.0), "padding_mode": "zeros", "diagonal": True},
        np.arange(24).reshape((1, 4, 6)),  # data
        {"affine": np.array([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), "mode": "nearest"},
        np.array(
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
        {"pixdim": (5.0, 3.0, 6.0), "padding_mode": "border", "diagonal": True, "dtype": np.float32},
        np.arange(24).reshape((1, 4, 6)),  # data
        {"affine": np.array([[-4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), "mode": "bilinear"},
        np.array(
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
        {"pixdim": (5.0, 3.0, 6.0), "padding_mode": "zeros", "diagonal": True, "dtype": np.float32},
        np.arange(24).reshape((1, 4, 6)),  # data
        {"affine": np.array([[-4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), "mode": "bilinear"},
        np.array(
            [
                [
                    [18.0000, 18.6000, 19.2000, 19.8000, 20.4000, 21.0000, 21.6000, 22.2000, 22.8000],
                    [10.5000, 11.1000, 11.7000, 12.3000, 12.9000, 13.5000, 14.1000, 14.7000, 15.3000],
                    [3.0000, 3.6000, 4.2000, 4.8000, 5.4000, 6.0000, 6.6000, 7.2000, 7.8000],
                ]
            ]
        ),
    ],
]


class TestSpacingCase(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_spacing(self, init_param, img, data_param, expected_output):
        res = Spacing(**init_param)(img, **data_param)
        np.testing.assert_allclose(res[0], expected_output, atol=1e-6)
        sr = len(res[0].shape) - 1
        if isinstance(init_param["pixdim"], float):
            init_param["pixdim"] = [init_param["pixdim"]] * sr
        init_pixdim = ensure_tuple(init_param["pixdim"])
        init_pixdim = init_param["pixdim"][:sr]
        np.testing.assert_allclose(init_pixdim[:sr], np.sqrt(np.sum(np.square(res[2]), axis=0))[:sr])

    def test_ill_pixdim(self):
        with self.assertRaises(ValueError):
            Spacing(pixdim=(-1, 2.0))(np.zeros((1, 1)))


if __name__ == "__main__":
    unittest.main()
