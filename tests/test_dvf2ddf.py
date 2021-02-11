import unittest

import numpy as np
import torch
from parameterized import parameterized
from torch import nn
from torch.optim import SGD

from monai.networks.blocks.warp import DVF2DDF

TEST_CASES = [
    [{"spatial_dims": 2, "num_steps": 1}, {"dvf": torch.zeros(1, 2, 2, 2)}, torch.zeros(1, 2, 2, 2)],
    [
        {"spatial_dims": 3, "num_steps": 1},
        {"dvf": torch.ones(1, 3, 2, 2, 2)},
        torch.tensor([[[1.0000, 0.7500], [0.7500, 0.6250]], [[0.7500, 0.6250], [0.6250, 0.5625]]])
        .reshape(1, 1, 2, 2, 2)
        .expand(-1, 3, -1, -1, -1),
    ],
    [
        {"spatial_dims": 3, "num_steps": 2},
        {"dvf": torch.ones(1, 3, 2, 2, 2)},
        torch.tensor([[[0.9175, 0.6618], [0.6618, 0.5306]], [[0.6618, 0.5306], [0.5306, 0.4506]]])
        .reshape(1, 1, 2, 2, 2)
        .expand(-1, 3, -1, -1, -1),
    ],
]


class TestDVF2DDF(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_value(self, input_param, input_data, expected_val):
        layer = DVF2DDF(**input_param)
        result = layer(**input_data)
        np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)

    def test_gradient(self):
        network = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1)
        dvf2ddf = DVF2DDF(spatial_dims=2, num_steps=1)
        optimizer = SGD(network.parameters(), lr=0.01)
        x = torch.ones((1, 1, 5, 5))
        x = network(x)
        x = dvf2ddf(x)
        loss = torch.sum(x)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
