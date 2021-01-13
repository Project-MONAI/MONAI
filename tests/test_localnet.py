import unittest
from itertools import product

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.localnet import LocalNet
from tests.utils import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"


param_variations_2d = {
    "spatial_dims": [2],
    "in_channels": [2],
    "out_channels": [2],
    "num_channel_initial": [16],
    "extract_levels": [[0, 1, 2]],
    "out_kernel_initializer": ["zeros", None],
    "out_activation": ["sigmoid", None],
}
TEST_CASE_LOCALNET_2D = [dict(zip(param_variations_2d, v)) for v in product(*param_variations_2d.values())]
TEST_CASE_LOCALNET_2D = [
    [input_param, (1, input_param["in_channels"], 16, 16), (1, input_param["out_channels"], 16, 16)]
    for input_param in TEST_CASE_LOCALNET_2D
]


param_variations_3d = {
    "spatial_dims": [3],
    "in_channels": [2, 3],
    "out_channels": [1, 3],
    "num_channel_initial": [16, 32],
    "extract_levels": [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]],
    "out_kernel_initializer": ["zeros", None],
    "out_activation": ["sigmoid", None],
}
TEST_CASE_LOCALNET_3D = [dict(zip(param_variations_3d, v)) for v in product(*param_variations_3d.values())]
TEST_CASE_LOCALNET_3D = [
    [input_param, (1, input_param["in_channels"], 16, 16, 16), (1, input_param["out_channels"], 16, 16, 16)]
    for input_param in TEST_CASE_LOCALNET_3D
]


class TestDynUNet(unittest.TestCase):
    @parameterized.expand(TEST_CASE_LOCALNET_2D + TEST_CASE_LOCALNET_3D)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = LocalNet(**input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_shape(self):
        with self.assertRaisesRegex(ValueError, ""):
            input_param, _, _ = TEST_CASE_LOCALNET_2D[0]
            input_shape = (1, input_param["in_channels"], 17, 17)
            net = LocalNet(**input_param).to(device)
            net.forward(torch.randn(input_shape).to(device))

    def test_script(self):
        input_param, input_shape, _ = TEST_CASE_LOCALNET_2D[0]
        net = LocalNet(**input_param)
        test_data = torch.randn(input_shape)
        test_script_save(net, test_data)


if __name__ == "__main__":
    unittest.main()
