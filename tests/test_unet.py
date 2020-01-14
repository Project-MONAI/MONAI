import unittest

import torch
from parameterized import parameterized

from monai.networks.nets.unet import UNet


class TestUNET(unittest.TestCase):

    @parameterized.expand([
        [
            {
                'dimensions': 2,
                'in_channels': 1,
                'num_classes': 3,
                'channels': (16, 32, 64),
                'strides': (2, 2),
                'num_res_units': 1,
            },
            torch.randn(16, 1, 32, 32),
            (16, 32, 32),
        ],
        [
            {
                'dimensions': 3,
                'in_channels': 1,
                'num_classes': 3,
                'channels': (16, 32, 64),
                'strides': (2, 2),
                'num_res_units': 1,
            },
            torch.randn(16, 1, 32, 32, 32),
            (16, 32, 32, 32),
        ],
    ])
    def test_shape(self, input_param, input_data, expected_shape):
        result = UNet(**input_param).forward(input_data)[1]
        self.assertEqual(result.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
