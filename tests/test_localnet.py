import unittest

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.localnet import LocalNet
from tests.utils import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"


TEST_CASE_LOCALNET_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 2,
            "num_channel_initial": 16,
            "out_kernel_initializer": "kaiming_uniform",
            "out_activation": None,
            "out_channels": 2,
            "extract_levels": (0, 1),
            "pooling": False,
            "concat_skip": True,
        },
        (1, 2, 16, 16),
        (1, 2, 16, 16),
    ]
]

TEST_CASE_LOCALNET_3D = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 2,
            "num_channel_initial": 16,
            "out_kernel_initializer": "zeros",
            "out_activation": "sigmoid",
            "out_channels": 2,
            "extract_levels": (0, 1, 2, 3),
            "pooling": True,
            "concat_skip": False,
        },
        (1, 2, 16, 16, 16),
        (1, 2, 16, 16, 16),
    ]
]


class TestLocalNet(unittest.TestCase):
    @parameterized.expand(TEST_CASE_LOCALNET_2D + TEST_CASE_LOCALNET_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = LocalNet(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        input_param, input_shape, _ = TEST_CASE_LOCALNET_2D[0]
        net = LocalNet(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
