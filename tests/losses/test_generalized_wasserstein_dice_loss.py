# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from monai.losses import GeneralizedWassersteinDiceLoss
from tests.test_utils import test_script_save


class TestGeneralizedWassersteinDiceLoss(unittest.TestCase):
    def test_bin_seg_2d(self):
        target = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])

        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)
        pred_very_good = 1000 * F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
        pred_very_poor = 1000 * F.one_hot(1 - target, num_classes=2).permute(0, 3, 1, 2).float()

        for weight_mode in ["default", "GDL"]:
            # initialize the loss
            loss = GeneralizedWassersteinDiceLoss(
                dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]), weighting_mode=weight_mode
            )

            # the loss for pred_very_good should be close to 0
            loss_good = float(loss.forward(pred_very_good, target))
            self.assertAlmostEqual(loss_good, 0.0, places=3)

            # same test, but with target with a class dimension
            target_4dim = target.unsqueeze(1)
            loss_good = float(loss.forward(pred_very_good, target_4dim))
            self.assertAlmostEqual(loss_good, 0.0, places=3)

            # the loss for pred_very_poor should be close to 1
            loss_poor = float(loss.forward(pred_very_poor, target))
            self.assertAlmostEqual(loss_poor, 1.0, places=3)

    def test_different_target_data_type(self):
        """
        Test if the loss is compatible with all the integer types
        for the target segmentation.
        """
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])

        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()

        target_uint8 = target.to(torch.uint8)
        target_int8 = target.to(torch.int8)
        target_short = target.short()
        target_int = target.int()
        target_long = target.long()
        target_list = [target_uint8, target_int8, target_short, target_int, target_long]

        for w_mode in ["default", "GDL"]:
            # initialize the loss
            loss = GeneralizedWassersteinDiceLoss(dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]), weighting_mode=w_mode)

            # The test should pass irrespectively of the integer type used
            for t in target_list:
                # the loss for pred_very_good should be close to 0
                loss_good = float(loss.forward(pred_very_good, t))
                self.assertAlmostEqual(loss_good, 0.0, places=3)

    def test_empty_class_2d(self):
        num_classes = 2
        target = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)
        pred_very_good = 1000 * F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        pred_very_poor = 1000 * F.one_hot(1 - target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        for w_mode in ["default", "GDL"]:
            # initialize the loss
            loss = GeneralizedWassersteinDiceLoss(dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]), weighting_mode=w_mode)

            # loss for pred_very_good should be close to 0
            loss_good = float(loss.forward(pred_very_good, target))
            self.assertAlmostEqual(loss_good, 0.0, places=3)

            # loss for pred_very_poor should be close to 1
            loss_poor = float(loss.forward(pred_very_poor, target))
            self.assertAlmostEqual(loss_poor, 1.0, places=3)

    def test_bin_seg_3d(self):
        # define 3d examples
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
        pred_very_poor = 1000 * F.one_hot(1 - target, num_classes=2).permute(0, 4, 1, 2, 3).float()

        for w_mode in ["default", "GDL"]:
            # initialize the loss
            loss = GeneralizedWassersteinDiceLoss(dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]), weighting_mode=w_mode)

            # mean dice loss for pred_very_good should be close to 0
            loss_good = float(loss.forward(pred_very_good, target))
            self.assertAlmostEqual(loss_good, 0.0, places=3)

            # mean dice loss for pred_very_poor should be close to 1
            loss_poor = float(loss.forward(pred_very_poor, target))
            self.assertAlmostEqual(loss_poor, 1.0, places=3)

    def test_convergence(self):
        """
        The goal of this test is to assess if the gradient of the loss function
        is correct by testing if we can train a one layer neural network
        to segment one image.
        We verify that the loss is decreasing in almost all SGD steps.
        """
        learning_rate = 0.001
        max_iter = 50

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
        image = 12 * target_seg + 27  # dummy image to segment
        image = image.float()
        num_classes = 2
        num_voxels = 3 * 4 * 4

        # define a model with one layer
        class OnelayerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(num_voxels, num_voxels * num_classes)

            def forward(self, x):
                x = x.view(-1, num_voxels)
                x = self.layer(x)
                x = x.view(-1, num_classes, 3, 4, 4)
                return x

        for w_mode in ["default", "GDL"]:
            # initialise the network
            net = OnelayerNet()

            # initialize the loss
            loss = GeneralizedWassersteinDiceLoss(dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]), weighting_mode=w_mode)

            # initialize an optimizer
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)

            # initial difference between pred and target
            pred_start = torch.argmax(net(image), dim=1)
            diff_start = torch.norm(pred_start.float() - target_seg.float())

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

            # difference between pred and target after training
            pred_end = torch.argmax(net(image), dim=1)
            diff_end = torch.norm(pred_end.float() - target_seg.float())

            # count the number of SGD steps in which the loss decreases
            num_decreasing_steps = 0
            for i in range(len(loss_history) - 1):
                if loss_history[i] > loss_history[i + 1]:
                    num_decreasing_steps += 1
            decreasing_steps_ratio = float(num_decreasing_steps) / (len(loss_history) - 1)

            # verify that the loss is decreasing for sufficiently many SGD steps
            self.assertTrue(decreasing_steps_ratio > 0.75)

            # check that the predicted segmentation has improved
            self.assertGreater(diff_start, diff_end)

    def test_batch_size_greater_than_one(self):
        """
        Regression test for https://github.com/Project-MONAI/MONAI/issues/4650
        With M=identity and batch_size > 1, the GWDL should produce the same
        per-sample loss values as with batch_size=1.
        """
        target_single = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        target_single = target_single.unsqueeze(0)  # shape (1, H, W)
        pred_single = 1000 * F.one_hot(target_single, num_classes=2).permute(0, 3, 1, 2).float()

        # Create a batch of size 2 by repeating the same sample
        target_batch = target_single.repeat(2, 1, 1)  # shape (2, H, W)
        pred_batch = pred_single.repeat(2, 1, 1, 1)  # shape (2, C, H, W)

        for w_mode in ["default", "GDL"]:
            loss_fn = GeneralizedWassersteinDiceLoss(
                dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]), weighting_mode=w_mode, reduction="none"
            )

            loss_single = loss_fn(pred_single, target_single)
            loss_batch = loss_fn(pred_batch, target_batch)

            # Each sample in the batch should produce the same loss as the single sample
            for i in range(2):
                self.assertAlmostEqual(
                    float(loss_batch[i]),
                    float(loss_single[0]),
                    places=5,
                    msg=f"Batch loss[{i}] != single loss for weighting_mode={w_mode}",
                )

        # Also test with mean reduction using a non-trivial (poor) prediction
        # so the expected loss is not near zero
        pred_poor = 1000 * F.one_hot(1 - target_single, num_classes=2).permute(0, 3, 1, 2).float()
        pred_poor_batch = pred_poor.repeat(2, 1, 1, 1)

        for w_mode in ["default", "GDL"]:
            loss_fn = GeneralizedWassersteinDiceLoss(
                dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]), weighting_mode=w_mode, reduction="mean"
            )

            loss_single = float(loss_fn(pred_poor, target_single))
            loss_batch = float(loss_fn(pred_poor_batch, target_batch))

            # Verify the loss is non-trivial (close to 1 for poor predictions)
            self.assertGreater(loss_single, 0.5, msg=f"Expected non-trivial loss for weighting_mode={w_mode}")
            self.assertAlmostEqual(
                loss_batch,
                loss_single,
                places=5,
                msg=f"Batch mean loss != single mean loss for weighting_mode={w_mode}",
            )

    def test_batch_size_different_samples(self):
        """
        Regression test for https://github.com/Project-MONAI/MONAI/issues/4650
        Verify loss is computed correctly when batch contains different samples.
        """
        target_a = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).unsqueeze(0)
        target_b = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).unsqueeze(0)

        pred_a = 1000 * F.one_hot(target_a, num_classes=2).permute(0, 3, 1, 2).float()
        # Use a poor prediction for sample b so its loss is non-trivial (~1.0)
        pred_b = 1000 * F.one_hot(1 - target_b, num_classes=2).permute(0, 3, 1, 2).float()

        # Combine into a batch
        target_batch = torch.cat([target_a, target_b], dim=0)
        pred_batch = torch.cat([pred_a, pred_b], dim=0)

        for w_mode in ["default", "GDL"]:
            loss_fn = GeneralizedWassersteinDiceLoss(
                dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]), weighting_mode=w_mode, reduction="none"
            )

            loss_a = float(loss_fn(pred_a, target_a))
            loss_b = float(loss_fn(pred_b, target_b))
            loss_batch = loss_fn(pred_batch, target_batch)

            self.assertAlmostEqual(
                float(loss_batch[0]), loss_a, places=5, msg=f"Batch loss[0] != loss_a for weighting_mode={w_mode}"
            )
            self.assertAlmostEqual(
                float(loss_batch[1]), loss_b, places=5, msg=f"Batch loss[1] != loss_b for weighting_mode={w_mode}"
            )

    def test_script(self):
        target = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])

        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)
        pred_very_good = 1000 * F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()

        loss = GeneralizedWassersteinDiceLoss(dist_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]), weighting_mode="default")

        test_script_save(loss, pred_very_good, target)


if __name__ == "__main__":
    unittest.main()
