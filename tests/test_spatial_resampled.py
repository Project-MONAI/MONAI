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

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.transforms.spatial.dictionary import SpatialResampled
from tests.utils import TEST_DEVICES, assert_allclose

TESTS = []

destinations_3d = [
    torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
    torch.tensor([[-1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
]
expected_3d = [
    torch.tensor([[[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], [[10.0, 11.0, 12.0], [7.0, 8.0, 9.0]]]]),
    torch.tensor([[[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]]),
]

for dst, expct in zip(destinations_3d, expected_3d):
    for device in TEST_DEVICES:
        for align in (True, False):
            for dtype in (torch.float32, torch.float64):
                interp = ("nearest", "bilinear")
                for interp_mode in interp:
                    for padding_mode in ("zeros", "border", "reflection"):
                        TESTS.append(
                            [
                                np.arange(12).reshape((1, 2, 2, 3)) + 1.0,  # data
                                *device,
                                dst,
                                {
                                    "dst_keys": "dst_affine",
                                    "dtype": dtype,
                                    "align_corners": align,
                                    "mode": interp_mode,
                                    "padding_mode": padding_mode,
                                },
                                expct,
                            ]
                        )

destinations_2d = [
    torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 1.0]]),  # flip the second
    torch.tensor([[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),  # flip the first
]

expected_2d = [torch.tensor([[[2.0, 1.0], [4.0, 3.0]]]), torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])]

for dst, expct in zip(destinations_2d, expected_2d):
    for device in TEST_DEVICES:
        for align in (False, True):
            for dtype in (torch.float32, torch.float64):
                for interp_mode in ("nearest", "bilinear"):
                    TESTS.append(
                        [
                            np.arange(4).reshape((1, 2, 2)) + 1.0,  # data
                            *device,
                            dst,
                            {
                                "dst_keys": "dst_affine",
                                "dtype": dtype,
                                "align_corners": align,
                                "mode": interp_mode,
                                "padding_mode": "zeros",
                            },
                            expct,
                        ]
                    )


class TestSpatialResample(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_flips_inverse(self, img, device, dst_affine, kwargs, expected_output):
        img = MetaTensor(img, affine=torch.eye(4)).to(device)
        data = {"img": img, "dst_affine": dst_affine}

        xform = SpatialResampled(keys="img", **kwargs)
        output_data = xform(data)
        out = output_data["img"]

        assert_allclose(out, expected_output, rtol=1e-2, atol=1e-2)
        assert_allclose(out.affine, dst_affine, rtol=1e-2, atol=1e-2)

        inverted = xform.inverse(output_data)["img"]
        self.assertEqual(inverted.applied_operations, [])  # no further invert after inverting
        expected_affine = to_affine_nd(len(out.affine) - 1, torch.eye(4))
        assert_allclose(inverted.affine, expected_affine, rtol=1e-2, atol=1e-2)
        assert_allclose(inverted, img, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
