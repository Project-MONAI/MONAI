# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import unittest
from monai.losses import FocalLoss


class TestFocalLoss(unittest.TestCase):
    def test_consistency_with_cross_entropy_2d(self):
        # For gamma=0 the focal loss reduces to the cross entropy loss
        focal_loss = FocalLoss(gamma=0.0, reduction="mean")
        ce = nn.CrossEntropyLoss(reduction="mean")
        max_error = 0
        class_num = 10
        batch_size = 128
        for _ in range(100):
            # Create a random tensor of shape (batch_size, class_num, 8, 4)
            x = torch.rand(batch_size, class_num, 8, 4, requires_grad=True)
            # Create a random batch of classes
            l = torch.randint(low=0, high=class_num, size=(batch_size, 8, 4))
            l = l.long()
            if torch.cuda.is_available():
                x = x.cuda()
                l = l.cuda()
            output0 = focal_loss.forward(x, l)
            output1 = ce.forward(x, l)
            a = float(output0.cpu().detach())
            b = float(output1.cpu().detach())
            if abs(a - b) > max_error:
                max_error = abs(a - b)
        self.assertAlmostEqual(max_error, 0.0, places=3)

    def test_consistency_with_cross_entropy_classification(self):
        # for gamma=0 the focal loss reduces to the cross entropy loss
        focal_loss = FocalLoss(gamma=0.0, reduction="mean")
        ce = nn.CrossEntropyLoss(reduction="mean")
        max_error = 0
        class_num = 10
        batch_size = 128
        for _ in range(100):
            # Create a random scores tensor of shape (batch_size, class_num)
            x = torch.rand(batch_size, class_num, requires_grad=True)
            # Create a random batch of classes
            l = torch.randint(low=0, high=class_num, size=(batch_size,))
            l = l.long()
            if torch.cuda.is_available():
                x = x.cuda()
                l = l.cuda()
            output0 = focal_loss.forward(x, l)
            output1 = ce.forward(x, l)
            a = float(output0.cpu().detach())
            b = float(output1.cpu().detach())
            if abs(a - b) > max_error:
                max_error = abs(a - b)
        self.assertAlmostEqual(max_error, 0.0, places=3)

    def test_bin_seg_2d(self):
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()

        # initialize the mean dice loss
        loss = FocalLoss()

        # focal loss for pred_very_good should be close to 0
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

        # Same test, but for target with a class dimension
        target = target.unsqueeze(1)  # shape (1, 1, H, W)
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_empty_class_2d(self):
        num_classes = 2
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # initialize the mean dice loss
        loss = FocalLoss()

        # focal loss for pred_very_good should be close to 0
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_multi_class_seg_2d(self):
        num_classes = 6  # labels 0 to 5
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # initialize the mean dice loss
        loss = FocalLoss()

        # focal loss for pred_very_good should be close to 0
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_bin_seg_3d(self):
        # define 2d examples
        target = torch.tensor(
            [
                # raw 0
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                # raw 1
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                # raw 2
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            ]
        )
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W, D)
        pred_very_good = 1000 * F.one_hot(target, num_classes=2).permute(0, 4, 1, 2, 3).float()

        # initialize the mean dice loss
        loss = FocalLoss()

        # focal loss for pred_very_good should be close to 0
        focal_loss_good = float(loss.forward(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_convergence(self):
        """
        The goal of this test is to assess if the gradient of the loss function
        is correct by testing if we can train a one layer neural network
        to segment one image.
        We verify that the loss is decreasing in almost all SGD steps.
        """
        learning_rate = 0.001
        max_iter = 20

        # define a simple 3d example
        target_seg = torch.tensor(
            [
                # raw 0
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                # raw 1
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
                # raw 2
                [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            ]
        )
        target_seg = torch.unsqueeze(target_seg, dim=0)
        image = 12 * target_seg + 27
        image = image.float()
        num_classes = 2
        num_voxels = 3 * 4 * 4

        # define a one layer model
        class OnelayerNet(nn.Module):
            def __init__(self):
                super(OnelayerNet, self).__init__()
                self.layer = nn.Linear(num_voxels, num_voxels * num_classes)

            def forward(self, x):
                x = x.view(-1, num_voxels)
                x = self.layer(x)
                x = x.view(-1, num_classes, 3, 4, 4)
                return x

        # initialise the network
        net = OnelayerNet()

        # initialize the loss
        loss = FocalLoss()

        # initialize an SGD
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

        loss_history = []
        # train the network
        for _ in range(max_iter):
            # set the gradient to zero
            optimizer.zero_grad()

            # forward pass
            output = net(image)
            loss_val = loss(output, target_seg)

            # backward pass
            loss_val.backward()
            optimizer.step()

            # stats
            loss_history.append(loss_val.item())

        # count the number of SGD steps in which the loss decreases
        num_decreasing_steps = 0
        for i in range(len(loss_history) - 1):
            if loss_history[i] > loss_history[i + 1]:
                num_decreasing_steps += 1
        decreasing_steps_ratio = float(num_decreasing_steps) / (len(loss_history) - 1)

        # verify that the loss is decreasing for sufficiently many SGD steps
        self.assertTrue(decreasing_steps_ratio > 0.9)


if __name__ == "__main__":
    unittest.main()
