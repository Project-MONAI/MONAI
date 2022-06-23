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
from monai.transforms import RandAffined
from monai.utils import GridSampleMode
from tests.utils import assert_allclose, is_tf32_env

_rtol = 1e-3 if is_tf32_env() else 1e-4

TESTS = []

for device in [None, "cpu", "cuda"] if torch.cuda.is_available() else [None, "cpu"]:
    TESTS.append(
        [
            dict(device=device, spatial_size=None, keys=("img", "seg")),
            {
                "img": MetaTensor(torch.arange(27).reshape((3, 3, 3))),
                "seg": MetaTensor(torch.arange(27).reshape((3, 3, 3))),
            },
            torch.arange(27).reshape((3, 3, 3)),
        ]
    )
    TESTS.append(
        [
            dict(device=device, spatial_size=(2, 2), keys=("img", "seg")),
            {"img": MetaTensor(torch.ones((3, 3, 3))), "seg": MetaTensor(torch.ones((3, 3, 3)))},
            torch.ones((3, 2, 2)),
        ]
    )
    TESTS.append(
        [
            dict(device=device, spatial_size=(2, 2), cache_grid=True, keys=("img", "seg")),
            {"img": MetaTensor(torch.ones((3, 3, 3))), "seg": MetaTensor(torch.ones((3, 3, 3)))},
            torch.ones((3, 2, 2)),
        ]
    )
    TESTS.append(
        [
            dict(device=device, spatial_size=(2, 2, 2), keys=("img", "seg")),
            {"img": MetaTensor(torch.ones((1, 3, 3, 3))), "seg": MetaTensor(torch.ones((1, 3, 3, 3)))},
            torch.ones((1, 2, 2, 2)),
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                spatial_size=(2, 2, 2),
                padding_mode="zeros",
                device=device,
                keys=("img", "seg"),
                mode="bilinear",
            ),
            {"img": MetaTensor(torch.ones((1, 3, 3, 3))), "seg": MetaTensor(torch.ones((1, 3, 3, 3)))},
            torch.tensor([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]]),
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                scale_range=[0.1, 0.2],
                spatial_size=(3, 3),
                keys=("img", "seg"),
                device=device,
            ),
            {
                "img": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
                "seg": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
            },
            torch.tensor([[[18.7362, 15.5820, 12.4278], [27.3988, 24.2446, 21.0904], [36.0614, 32.9072, 29.7530]]]),
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                mode=("bilinear", "nearest"),
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                scale_range=[0.1, 0.2],
                spatial_size=(3, 3),
                keys=("img", "seg"),
                device=device,
            ),
            {
                "img": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
                "seg": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
            },
            {
                "img": MetaTensor(
                    torch.tensor(
                        [
                            [
                                [18.736153, 15.581954, 12.4277525],
                                [27.398798, 24.244598, 21.090399],
                                [36.061443, 32.90724, 29.753046],
                            ]
                        ]
                    )
                ),
                "seg": MetaTensor(torch.tensor([[[19.0, 20.0, 12.0], [27.0, 28.0, 20.0], [35.0, 36.0, 29.0]]])),
            },
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                spatial_size=(2, 2, 2),
                padding_mode="zeros",
                device=device,
                keys=("img", "seg"),
                mode=GridSampleMode.BILINEAR,
            ),
            {"img": MetaTensor(torch.ones((1, 3, 3, 3))), "seg": MetaTensor(torch.ones((1, 3, 3, 3)))},
            torch.tensor([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]]),
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                mode=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                scale_range=[0.1, 0.2],
                spatial_size=(3, 3),
                keys=("img", "seg"),
                device=device,
            ),
            {
                "img": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
                "seg": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
            },
            {
                "img": MetaTensor(
                    np.array(
                        [
                            [
                                [18.736153, 15.581954, 12.4277525],
                                [27.398798, 24.244598, 21.090399],
                                [36.061443, 32.90724, 29.753046],
                            ]
                        ]
                    )
                ),
                "seg": MetaTensor(np.array([[[19.0, 20.0, 12.0], [27.0, 28.0, 20.0], [35.0, 36.0, 29.0]]])),
            },
        ]
    )
    TESTS.append(
        [
            dict(
                prob=0.9,
                mode=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
                rotate_range=(np.pi / 2,),
                shear_range=[1, 2],
                translate_range=[2, 1],
                scale_range=[0.1, 0.2],
                spatial_size=(3, 3),
                cache_grid=True,
                keys=("img", "seg"),
                device=device,
            ),
            {
                "img": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
                "seg": MetaTensor(torch.arange(64).reshape((1, 8, 8))),
            },
            {
                "img": MetaTensor(
                    torch.tensor(
                        [
                            [
                                [18.736153, 15.581954, 12.4277525],
                                [27.398798, 24.244598, 21.090399],
                                [36.061443, 32.90724, 29.753046],
                            ]
                        ]
                    )
                ),
                "seg": MetaTensor(torch.tensor([[[19.0, 20.0, 12.0], [27.0, 28.0, 20.0], [35.0, 36.0, 29.0]]])),
            },
        ]
    )


class TestRandAffined(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_affined(self, input_param, input_data, expected_val):
        g = RandAffined(**input_param).set_random_state(123)
        res = g(input_data)
        if input_param.get("cache_grid", False):
            self.assertTrue(g.rand_affine._cached_grid is not None)
        for key in res:
            result = res[key]
            self.assertIsInstance(result, MetaTensor)
            self.assertEqual(len(result.applied_operations), 1)
            expected = expected_val[key] if isinstance(expected_val, dict) else expected_val
            assert_allclose(result, expected, rtol=_rtol, atol=1e-3, type_test=False)

        g.set_random_state(4)
        res = g(input_data)

        # affine should be tensor because the resampler only supports pytorch backend
        if isinstance(res["img"], MetaTensor) and "extra_info" in res["img"].applied_operations[0]:
            if not res["img"].applied_operations[-1]["extra_info"]["do_resampling"]:
                return
            affine_img = res["img"].applied_operations[0]["extra_info"]["rand_affine_info"]["extra_info"]["affine"]
            affine_seg = res["seg"].applied_operations[0]["extra_info"]["rand_affine_info"]["extra_info"]["affine"]
            assert_allclose(affine_img, affine_seg, rtol=_rtol, atol=1e-3)

        res_inv = g.inverse(res)
        for k, v in res_inv.items():
            self.assertIsInstance(v, MetaTensor)
            self.assertEqual(len(v.applied_operations), 0)
            self.assertTupleEqual(v.shape, input_data[k].shape)

    def test_ill_cache(self):
        with self.assertWarns(UserWarning):
            # spatial size is None
            RandAffined(device=device, spatial_size=None, prob=1.0, cache_grid=True, keys=("img", "seg"))
        with self.assertWarns(UserWarning):
            # spatial size is dynamic
            RandAffined(device=device, spatial_size=(2, -1), prob=1.0, cache_grid=True, keys=("img", "seg"))


if __name__ == "__main__":
    unittest.main()
