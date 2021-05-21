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
import torch
from parameterized import parameterized

from monai.transforms import RandAffined
from monai.utils import GridSampleMode

TEST_CASES = [
    [
        dict(as_tensor_output=False, device=None, spatial_size=None, keys=("img", "seg")),
        {"img": torch.arange(27).reshape((3, 3, 3)), "seg": torch.arange(27).reshape((3, 3, 3))},
        np.arange(27).reshape((3, 3, 3)),
    ],
    [
        dict(as_tensor_output=False, device=None, spatial_size=(2, 2), keys=("img", "seg")),
        {"img": torch.ones((3, 3, 3)), "seg": torch.ones((3, 3, 3))},
        np.ones((3, 2, 2)),
    ],
    [
        dict(as_tensor_output=False, device=None, spatial_size=(2, 2), cache_grid=True, keys=("img", "seg")),
        {"img": torch.ones((3, 3, 3)), "seg": torch.ones((3, 3, 3))},
        np.ones((3, 2, 2)),
    ],
    [
        dict(as_tensor_output=True, device=None, spatial_size=(2, 2, 2), keys=("img", "seg")),
        {"img": torch.ones((1, 3, 3, 3)), "seg": torch.ones((1, 3, 3, 3))},
        torch.ones((1, 2, 2, 2)),
    ],
    [
        dict(
            prob=0.9,
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            as_tensor_output=True,
            spatial_size=(2, 2, 2),
            padding_mode="zeros",
            device=None,
            keys=("img", "seg"),
            mode="bilinear",
        ),
        {"img": torch.ones((1, 3, 3, 3)), "seg": torch.ones((1, 3, 3, 3))},
        torch.tensor([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]]),
    ],
    [
        dict(
            prob=0.9,
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            as_tensor_output=False,
            spatial_size=(2, 2, 2),
            padding_mode="zeros",
            device=None,
            cache_grid=True,
            keys=("img", "seg"),
            mode="bilinear",
        ),
        {"img": torch.ones((1, 3, 3, 3)), "seg": torch.ones((1, 3, 3, 3))},
        np.array([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]]),
    ],
    [
        dict(
            prob=0.9,
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            scale_range=[0.1, 0.2],
            as_tensor_output=True,
            spatial_size=(3, 3),
            keys=("img", "seg"),
            device=None,
        ),
        {"img": torch.arange(64).reshape((1, 8, 8)), "seg": torch.arange(64).reshape((1, 8, 8))},
        torch.tensor([[[18.7362, 15.5820, 12.4278], [27.3988, 24.2446, 21.0904], [36.0614, 32.9072, 29.7530]]]),
    ],
    [
        dict(
            prob=0.9,
            mode=("bilinear", "nearest"),
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            scale_range=[0.1, 0.2],
            as_tensor_output=False,
            spatial_size=(3, 3),
            keys=("img", "seg"),
            device=torch.device("cpu:0"),
        ),
        {"img": torch.arange(64).reshape((1, 8, 8)), "seg": torch.arange(64).reshape((1, 8, 8))},
        {
            "img": np.array(
                [
                    [
                        [18.736153, 15.581954, 12.4277525],
                        [27.398798, 24.244598, 21.090399],
                        [36.061443, 32.90724, 29.753046],
                    ]
                ]
            ),
            "seg": np.array([[[19.0, 20.0, 12.0], [27.0, 28.0, 20.0], [35.0, 36.0, 29.0]]]),
        },
    ],
    [
        dict(
            prob=0.9,
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            as_tensor_output=True,
            spatial_size=(2, 2, 2),
            padding_mode="zeros",
            device=None,
            keys=("img", "seg"),
            mode=GridSampleMode.BILINEAR,
        ),
        {"img": torch.ones((1, 3, 3, 3)), "seg": torch.ones((1, 3, 3, 3))},
        torch.tensor([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]]),
    ],
    [
        dict(
            prob=0.9,
            mode=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            scale_range=[0.1, 0.2],
            as_tensor_output=False,
            spatial_size=(3, 3),
            keys=("img", "seg"),
            device=torch.device("cpu:0"),
        ),
        {"img": torch.arange(64).reshape((1, 8, 8)), "seg": torch.arange(64).reshape((1, 8, 8))},
        {
            "img": np.array(
                [
                    [
                        [18.736153, 15.581954, 12.4277525],
                        [27.398798, 24.244598, 21.090399],
                        [36.061443, 32.90724, 29.753046],
                    ]
                ]
            ),
            "seg": np.array([[[19.0, 20.0, 12.0], [27.0, 28.0, 20.0], [35.0, 36.0, 29.0]]]),
        },
    ],
    [
        dict(
            prob=0.9,
            mode=(GridSampleMode.BILINEAR, GridSampleMode.NEAREST),
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            scale_range=[0.1, 0.2],
            as_tensor_output=False,
            spatial_size=(3, 3),
            cache_grid=True,
            keys=("img", "seg"),
            device=torch.device("cpu:0"),
        ),
        {"img": torch.arange(64).reshape((1, 8, 8)), "seg": torch.arange(64).reshape((1, 8, 8))},
        {
            "img": np.array(
                [
                    [
                        [18.736153, 15.581954, 12.4277525],
                        [27.398798, 24.244598, 21.090399],
                        [36.061443, 32.90724, 29.753046],
                    ]
                ]
            ),
            "seg": np.array([[[19.0, 20.0, 12.0], [27.0, 28.0, 20.0], [35.0, 36.0, 29.0]]]),
        },
    ],
]


class TestRandAffined(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_rand_affined(self, input_param, input_data, expected_val):
        g = RandAffined(**input_param).set_random_state(123)
        res = g(input_data)
        for key in res:
            result = res[key]
            if "_transforms" in key:
                continue
            expected = expected_val[key] if isinstance(expected_val, dict) else expected_val
            self.assertEqual(isinstance(result, torch.Tensor), isinstance(expected, torch.Tensor))
            if isinstance(result, torch.Tensor):
                np.testing.assert_allclose(result.cpu().numpy(), expected.cpu().numpy(), rtol=1e-4, atol=1e-4)
            else:
                np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_ill_cache(self):
        with self.assertWarns(UserWarning):
            # spatial size is None
            RandAffined(
                as_tensor_output=False, device=None, spatial_size=None, prob=1.0, cache_grid=True, keys=("img", "seg")
            )
        with self.assertWarns(UserWarning):
            # spatial size is dynamic
            RandAffined(
                as_tensor_output=False,
                device=None,
                spatial_size=(2, -1),
                prob=1.0,
                cache_grid=True,
                keys=("img", "seg"),
            )


if __name__ == "__main__":
    unittest.main()
