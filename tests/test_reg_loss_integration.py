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

import torch
import torch.nn as nn
import torch.optim as optim
from parameterized import parameterized

from monai.losses import BendingEnergyLoss, GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss

TEST_CASES = [
    [BendingEnergyLoss, {}, ["pred"]],
    [
        LocalNormalizedCrossCorrelationLoss,
        {"in_channels": 1, "kernel_size": 7, "kernel_type": "rectangular"},
        ["pred", "target"],
    ],
    [
        LocalNormalizedCrossCorrelationLoss,
        {"in_channels": 1, "kernel_size": 5, "kernel_type": "triangular"},
        ["pred", "target"],
    ],
    [
        LocalNormalizedCrossCorrelationLoss,
        {"in_channels": 1, "kernel_size": 3, "kernel_type": "gaussian"},
        ["pred", "target"],
    ],
    [GlobalMutualInformationLoss, {"num_bins": 10}, ["pred", "target"]],
]


class TestRegLossIntegration(unittest.TestCase):
    def setUp(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")

    def tearDown(self):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    @parameterized.expand(TEST_CASES)
    def test_convergence(self, loss_type, loss_args, forward_args):
        """
        The goal of this test is to assess if the gradient of the loss function
        is correct by testing if we can train a one layer neural network
        to segment one image.
        We verify that the loss is decreasing in almost all SGD steps.
        """
        learning_rate = 0.001
        max_iter = 100

        # define a simple 3d example
        target = torch.rand((1, 1, 5, 5, 5), device=self.device)
        image = 12 * target + 27
        image = image.to(device=self.device)

        # define a one layer model
        class OnelayerNet(nn.Module):
            def __init__(self):
                super(OnelayerNet, self).__init__()
                self.layer = nn.Sequential(
                    nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
                )

            def forward(self, x):
                return self.layer(x)

        # initialise the network
        net = OnelayerNet().to(self.device)

        # initialize the loss
        loss = loss_type(**loss_args).to(self.device)

        # initialize a SGD optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        # train the network
        for it in range(max_iter):
            # set the gradient to zero
            optimizer.zero_grad()

            # forward pass
            output = net(image)
            loss_input = {"pred": output, "target": target}

            loss_val = loss(**{k: loss_input[k] for k in forward_args})
            if it == 0:
                init_loss = loss_val

            # backward pass
            loss_val.backward()
            optimizer.step()
        self.assertTrue(init_loss > loss_val, "loss did not decrease")


if __name__ == "__main__":
    unittest.main()
