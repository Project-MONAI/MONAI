import unittest

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.localnet import LocalNet
from tests.utils import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_LOCALNET_2D = []
for out_kernel_initializer in ["zeros", None]:
    for out_activation in [None]:
        spatial_dims = 2
        in_channels = 2
        out_channels = 2
        num_channel_initial = 16
        extract_levels = [0, 1, 2]
        in_size = 32
        input_shape = (1, in_channels, *[in_size] * spatial_dims)
        expected_shape = (1, out_channels, *[in_size] * spatial_dims)
        test_case = [
            {
                "spatial_dims": spatial_dims,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_channel_initial": num_channel_initial,
                "extract_levels": extract_levels,
                "out_kernel_initializer": out_kernel_initializer,
                "out_activation": out_activation
            },
            input_shape,
            expected_shape
        ]
        TEST_CASE_LOCALNET_2D.append(test_case)


TEST_CASE_LOCALNET_3D = []
for in_channels in [2, 3]:
    for out_channels in [1, 3]:
        for num_channel_initial in [16, 32]:
            for extract_levels in [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]:
                for out_kernel_initializer in ["zeros", None]:
                    for out_activation in [None]:
                        spatial_dims = 3
                        in_size = 32
                        input_shape = (1, in_channels, *[in_size] * spatial_dims)
                        expected_shape = (1, out_channels, *[in_size] * spatial_dims)
                        test_case = [
                            {
                                "spatial_dims": spatial_dims,
                                "in_channels": in_channels,
                                "out_channels": out_channels,
                                "num_channel_initial": num_channel_initial,
                                "extract_levels": extract_levels,
                                "out_kernel_initializer": out_kernel_initializer,
                                "out_activation": out_activation
                            },
                            input_shape,
                            expected_shape
                        ]
                        TEST_CASE_LOCALNET_3D.append(test_case)


class TestDynUNet(unittest.TestCase):
    @parameterized.expand(TEST_CASE_LOCALNET_2D)
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