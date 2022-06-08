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

from monai.data import MetaTensor
from monai.transforms import Affine
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = []
for p in TEST_NDARRAYS:
    for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(9).reshape((1, 3, 3))), "spatial_size": (-1, 0)},
                p(np.arange(9).reshape(1, 3, 3)),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device, image_only=True),
                {"img": p(np.arange(9).reshape((1, 3, 3))), "spatial_size": (-1, 0)},
                p(np.arange(9).reshape(1, 3, 3)),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(4).reshape((1, 2, 2)))},
                p(np.arange(4).reshape(1, 2, 2)),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(4).reshape((1, 2, 2))), "spatial_size": (4, 4)},
                p(np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])),
            ]
        )
        TESTS.append(
            [
                dict(rotate_params=[np.pi / 2], padding_mode="zeros", device=device),
                {"img": p(np.arange(4).reshape((1, 2, 2))), "spatial_size": (4, 4)},
                p(np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])),
            ]
        )
        TESTS.append(
            [
                dict(
                    affine=p(torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
                    padding_mode="zeros",
                    device=device,
                ),
                {"img": p(np.arange(4).reshape((1, 2, 2))), "spatial_size": (4, 4)},
                p(np.array([[[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(27).reshape((1, 3, 3, 3))), "spatial_size": (-1, 0, 0)},
                p(np.arange(27).reshape(1, 3, 3, 3)),
            ]
        )
        TESTS.append(
            [
                dict(padding_mode="zeros", device=device),
                {"img": p(np.arange(8).reshape((1, 2, 2, 2))), "spatial_size": (4, 4, 4)},
                p(
                    np.array(
                        [
                            [
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 2.0, 3.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 4.0, 5.0, 0.0],
                                    [0.0, 6.0, 7.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                            ]
                        ]
                    )
                ),
            ]
        )
        TESTS.append(
            [
                dict(rotate_params=[np.pi / 2], padding_mode="zeros", device=device),
                {"img": p(np.arange(8).reshape((1, 2, 2, 2))), "spatial_size": (4, 4, 4)},
                p(
                    np.array(
                        [
                            [
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 2.0, 0.0, 0.0],
                                    [0.0, 3.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 6.0, 4.0, 0.0],
                                    [0.0, 7.0, 5.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                                [
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                ],
                            ]
                        ]
                    )
                ),
            ]
        )


class TestAffine(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_affine(self, input_param, input_data, expected_val):
        g = Affine(**input_param)
        result = g(**input_data)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(input_data["img"], MetaTensor):
            im_inv = g.inverse(result)
            self.assertTrue(not im_inv.applied_operations)
            assert_allclose(im_inv.shape, input_data["img"].shape)
            assert_allclose(im_inv.affine, input_data["img"].affine, atol=1e-3, rtol=1e-3)
        assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4, type_test=False)


if __name__ == "__main__":
    unittest.main()
