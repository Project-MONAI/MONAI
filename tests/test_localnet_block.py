import unittest
from itertools import product

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.localnet_block import ExtractBlock, LocalNetDownSampleBlock, LocalNetUpSampleBlock

TEST_CASE_DOWN_SAMPLE = [
    [{"spatial_dims": spatial_dims, "in_channels": 2, "out_channels": 4, "kernel_size": 3}] for spatial_dims in [2, 3]
]

TEST_CASE_UP_SAMPLE = [[{"spatial_dims": spatial_dims, "in_channels": 4, "out_channels": 2}] for spatial_dims in [2, 3]]

extract_param_option = {
    "spatial_dims": [2, 3],
    "in_channels": [2],
    "out_channels": [3],
    "act": ["sigmoid", None],
    "kernel_initializer": ["zeros", None],
}
TEST_CASE_EXTRACT = [dict(zip(extract_param_option, v)) for v in product(*extract_param_option.values())]
TEST_CASE_EXTRACT = [[i] for i in TEST_CASE_EXTRACT]

in_size = 4


class TestLocalNetDownSampleBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_DOWN_SAMPLE)
    def test_shape(self, input_param):
        net = LocalNetDownSampleBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([in_size] * input_param["spatial_dims"]))
        expect_mid_shape = (1, input_param["out_channels"], *([in_size] * input_param["spatial_dims"]))
        expect_x_shape = (1, input_param["out_channels"], *([in_size / 2] * input_param["spatial_dims"]))
        with eval_mode(net):
            x, mid = net(torch.randn(input_shape))
            self.assertEqual(x.shape, expect_x_shape)
            self.assertEqual(mid.shape, expect_mid_shape)

    def test_ill_arg(self):
        # even kernel_size
        with self.assertRaises(NotImplementedError):
            LocalNetDownSampleBlock(spatial_dims=2, in_channels=2, out_channels=4, kernel_size=4)

    @parameterized.expand(TEST_CASE_DOWN_SAMPLE)
    def test_ill_shape(self, input_param):
        net = LocalNetDownSampleBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([5] * input_param["spatial_dims"]))
        with self.assertRaises(ValueError):
            with eval_mode(net):
                net(torch.randn(input_shape))


class TestLocalNetUpSampleBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_UP_SAMPLE)
    def test_shape(self, input_param):
        net = LocalNetUpSampleBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([in_size] * input_param["spatial_dims"]))
        mid_shape = (1, input_param["out_channels"], *([in_size * 2] * input_param["spatial_dims"]))
        expected_shape = mid_shape
        with eval_mode(net):
            result = net(torch.randn(input_shape), torch.randn(mid_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        # channel unmatch
        with self.assertRaises(ValueError):
            LocalNetUpSampleBlock(spatial_dims=2, in_channels=2, out_channels=2)

    @parameterized.expand(TEST_CASE_UP_SAMPLE)
    def test_ill_shape(self, input_param):
        net = LocalNetUpSampleBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([in_size] * input_param["spatial_dims"]))
        mid_shape = (1, input_param["out_channels"], *([in_size] * input_param["spatial_dims"]))
        with self.assertRaises(ValueError):
            with eval_mode(net):
                net(torch.randn(input_shape), torch.randn(mid_shape))


class TestExtractBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_EXTRACT)
    def test_shape(self, input_param):
        net = ExtractBlock(**input_param)
        input_shape = (1, input_param["in_channels"], *([in_size] * input_param["spatial_dims"]))
        expected_shape = (1, input_param["out_channels"], *([in_size] * input_param["spatial_dims"]))
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
