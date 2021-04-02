import unittest

import numpy as np
import torch
from parameterized import parameterized
from torch.autograd import gradcheck

from monai.config.deviceconfig import USE_COMPILED
from monai.networks.blocks.warp import Warp
from monai.utils import GridSampleMode, GridSamplePadMode

LOW_POWER_TEST_CASES = [  # run with BUILD_MONAI=1 to test csrc/resample, BUILD_MONAI=0 to test native grid_sample
    [
        {"mode": "nearest", "padding_mode": "zeros"},
        {"image": torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), "ddf": torch.zeros(1, 2, 2, 2)},
        torch.arange(4).reshape((1, 1, 2, 2)),
    ],
    [
        {"mode": "bilinear", "padding_mode": "zeros"},
        {"image": torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), "ddf": torch.ones(1, 2, 2, 2)},
        torch.tensor([[[[3, 0], [0, 0]]]]),
    ],
    [
        {"mode": "bilinear", "padding_mode": "border"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]]),
    ],
    [
        {"mode": "bilinear", "padding_mode": "reflection"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[7.0, 6.0], [5.0, 4.0]], [[3.0, 2.0], [1.0, 0.0]]]]]),
    ],
]

CPP_TEST_CASES = [  # high order, BUILD_MONAI=1 to test csrc/resample
    [
        {"mode": 2, "padding_mode": "border"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[0.0000, 0.1250], [0.2500, 0.3750]], [[0.5000, 0.6250], [0.7500, 0.8750]]]]]),
    ],
    [
        {"mode": 2, "padding_mode": "reflection"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[5.2500, 4.7500], [4.2500, 3.7500]], [[3.2500, 2.7500], [2.2500, 1.7500]]]]]),
    ],
    [
        {"mode": 2, "padding_mode": "zeros"},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[0.0000, 0.0020], [0.0039, 0.0410]], [[0.0078, 0.0684], [0.0820, 0.6699]]]]]),
    ],
    [
        {"mode": 2, "padding_mode": 7},
        {
            "image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float),
            "ddf": torch.ones(1, 3, 2, 2, 2) * -1,
        },
        torch.tensor([[[[[0.0000, 0.0020], [0.0039, 0.0410]], [[0.0078, 0.0684], [0.0820, 0.6699]]]]]),
    ],
    [
        {"mode": 3, "padding_mode": "reflection"},
        {"image": torch.arange(8).reshape((1, 1, 2, 2, 2)).to(dtype=torch.float), "ddf": torch.ones(1, 3, 2, 2, 2)},
        torch.tensor([[[[[4.6667, 4.3333], [4.0000, 3.6667]], [[3.3333, 3.0000], [2.6667, 2.3333]]]]]),
    ],
]

TEST_CASES = LOW_POWER_TEST_CASES
if USE_COMPILED:
    TEST_CASES += CPP_TEST_CASES


class TestWarp(unittest.TestCase):
    @parameterized.expand(TEST_CASES, skip_on_empty=True)
    def test_resample(self, input_param, input_data, expected_val):
        warp_layer = Warp(**input_param)
        result = warp_layer(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)

    def test_ill_shape(self):
        warp_layer = Warp()
        with self.assertRaisesRegex(ValueError, ""):
            warp_layer(
                image=torch.arange(4).reshape((1, 1, 1, 2, 2)).to(dtype=torch.float), ddf=torch.zeros(1, 2, 2, 2)
            )
        with self.assertRaisesRegex(ValueError, ""):
            warp_layer(
                image=torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), ddf=torch.zeros(1, 2, 1, 2, 2)
            )
        with self.assertRaisesRegex(ValueError, ""):
            warp_layer(image=torch.arange(4).reshape((1, 1, 2, 2)).to(dtype=torch.float), ddf=torch.zeros(1, 2, 3, 3))

    def test_grad(self):
        for b in GridSampleMode:
            for p in GridSamplePadMode:
                warp_layer = Warp(mode=b.value, padding_mode=p.value)
                input_image = torch.rand((2, 3, 20, 20), dtype=torch.float64) * 10.0
                ddf = torch.rand((2, 2, 20, 20), dtype=torch.float64) * 2.0
                input_image.requires_grad = True
                ddf.requires_grad = False  # Jacobian mismatch for output 0 with respect to input 1
                gradcheck(warp_layer, (input_image, ddf), atol=1e-2, eps=1e-2)


if __name__ == "__main__":
    unittest.main()
