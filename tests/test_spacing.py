# Copyright 2020 MONAI Consortium
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

TEST_CASES = [
    [{"pixdim": (2.0,), "mode": "constant"}, np.ones((1, 2)), {"affine": np.eye(4)}, np.array([[1.0, 0.0]])],  # data
    [
        {"pixdim": (1.0, 0.2, 1.5), "mode": "constant"},
        np.ones((1, 2, 1, 2)),  # data
        {"affine": np.eye(4)},
        np.array([[[[1.0, 0.0]], [[1.0, 0.0]]]]),
    ],
    [
        {"pixdim": (1.0, 0.2, 1.5), "diagonal": False, "mode": "constant"},
        np.ones((1, 2, 1, 2)),  # data
        {"affine": np.array([[2, 1, 0, 4], [-1, -3, 0, 5], [0, 0, 2.0, 5], [0, 0, 0, 1]])},
        np.zeros((1, 3, 1, 2)),
    ],
    [
        {"pixdim": (3.0, 1.0), "mode": "constant"},
        np.arange(24).reshape((2, 3, 4)),  # data
        {"affine": np.diag([-3.0, 0.2, 1.5, 1])},
        np.array([[[0, 0], [4, 0], [8, 0]], [[12, 0], [16, 0], [20, 0]]]),
    ],
    [
        {"pixdim": (3.0, 1.0), "mode": "constant"},
        np.arange(24).reshape((2, 3, 4)),  # data
        {},
        np.array([[[0, 1, 2, 3], [0, 0, 0, 0]], [[12, 13, 14, 15], [0, 0, 0, 0]]]),
    ],
    [
        {"pixdim": (1.0, 1.0), "mode": "constant"},
        np.arange(24).reshape((2, 3, 4)),  # data
        {},
        np.array(
            [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        ),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "mode": "constant"},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {"affine": np.array([[-4, 0, 0, 4], [0, 5, 0, -5], [0, 0, 6, -6], [0, 0, 0, 1]])},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "diagonal": True, "mode": "constant"},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {"affine": np.array([[-4, 0, 0, 4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]])},
        np.array(
            [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
        ),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "mode": "nearest", "diagonal": True},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {"affine": np.array([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]])},
        np.array(
            [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
        ),
    ],
    [
        {"pixdim": (4.0, 5.0, 6.0), "mode": "nearest", "diagonal": True},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {"affine": np.array([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), "interp_order": 0},
        np.array(
            [[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]]
        ),
    ],
    [
        {"pixdim": (2.0, 5.0, 6.0), "mode": "constant", "diagonal": True},
        np.arange(24).reshape((1, 4, 6)),  # data
        {"affine": np.array([[-4, 0, 0, -4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), "interp_order": 0},
        np.array(
            [
                [
                    [18, 19, 20, 21, 22, 23],
                    [18, 19, 20, 21, 22, 23],
                    [12, 13, 14, 15, 16, 17],
                    [12, 13, 14, 15, 16, 17],
                    [6, 7, 8, 9, 10, 11],
                    [6, 7, 8, 9, 10, 11],
                    [0, 1, 2, 3, 4, 5],
                ]
            ]
        ),
    ],
    [
        {"pixdim": (5.0, 3.0, 6.0), "mode": "constant", "diagonal": True, "dtype": np.float32},
        np.arange(24).reshape((1, 4, 6)),  # data
        {"affine": np.array([[-4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), "interp_order": 0},
        np.array(
            [
                [
                    [18.0, 19.0, 19.0, 20.0, 20.0, 21.0, 22.0, 22.0, 23],
                    [12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 16.0, 16.0, 17.0],
                    [6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 10.0, 11.0],
                ]
            ],
        ),
    ],
    [
        {"pixdim": (5.0, 3.0, 6.0), "mode": "constant", "diagonal": True, "dtype": np.float32},
        np.arange(24).reshape((1, 4, 6)),  # data
        {"affine": np.array([[-4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), "interp_order": 2},
        np.array(
            [
                [
                    [18.0, 18.492683, 19.22439, 19.80683, 20.398048, 21.0, 21.570732, 22.243902, 22.943415],
                    [10.392858, 10.88554, 11.617248, 12.199686, 12.790906, 13.392858, 13.963589, 14.63676, 15.336272],
                    [2.142857, 2.63554, 3.3672473, 3.9496865, 4.540906, 5.142857, 5.7135887, 6.3867598, 7.086272],
                ]
            ],
        ),
    ],
]


class TestSpacingCase(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_spacing(self, init_param, img, data_param, expected_output):
        res = Spacing(**init_param)(img, **data_param)
        np.testing.assert_allclose(res[0], expected_output, atol=1e-6)
        if "original_affine" in data_param:
            np.testing.assert_allclose(res[1], data_param["original_affine"])
        np.testing.assert_allclose(
            init_param["pixdim"], np.sqrt(np.sum(np.square(res[2]), axis=0))[: len(init_param["pixdim"])]
        )


if __name__ == "__main__":
    unittest.main()
