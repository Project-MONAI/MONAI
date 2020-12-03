import unittest

import torch
from parameterized import parameterized

from monai.networks.layers import Act
from monai.networks.nets import AutoEncoder
from tests.utils import test_script_save

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TEST_CASE_0 = [  # single channel 2D, batch 4, no residual
    {
        "dimensions": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
        "num_res_units": 0,
    },
    (1, 1, 128, 128),
    (1, 1, 128, 128),
]

TEST_CASE_1 = [  # single channel 2D, batch 4
    {
        "dimensions": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
    },
    (1, 1, 128, 128),
    (1, 1, 128, 128),
]

TEST_CASE_2 = [  # 3-channel 2D, batch 4, LeakyReLU activation
    {
        "dimensions": 2,
        "in_channels": 3,
        "out_channels": 3,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
        "act": (Act.LEAKYRELU, {"negative_slope": 0.2}),
    },
    (1, 3, 128, 128),
    (1, 3, 128, 128),
]

TEST_CASE_3 = [  # 4-channel 3D, batch 4
    {
        "dimensions": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (4, 8, 16),
        "strides": (2, 2, 2),
    },
    (1, 4, 128, 128, 128),
    (1, 3, 128, 128, 128),
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


class TestAutoEncoder(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = AutoEncoder(**input_param).to(device)
        net.eval()
        with torch.no_grad():
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = AutoEncoder(dimensions=2, in_channels=1, out_channels=1, channels=(4, 8), strides=(2, 2))
        test_data = torch.randn(2, 1, 32, 32)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
