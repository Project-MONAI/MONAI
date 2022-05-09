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

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import FocalLoss
from monai.networks import one_hot
from tests.utils import test_script_save


class TestFocalLoss(unittest.TestCase):
    def test_consistency_with_cross_entropy_2d(self):
        """For gamma=0 the focal loss reduces to the cross entropy loss"""
        focal_loss = FocalLoss(to_onehot_y=False, gamma=0.0, reduction="mean", weight=1.0)
        ce = nn.BCEWithLogitsLoss(reduction="mean")
        max_error = 0
        class_num = 10
        batch_size = 128
        for _ in range(100):
            # Create a random tensor of shape (batch_size, class_num, 8, 4)
            x = torch.rand(batch_size, class_num, 8, 4, requires_grad=True)
            # Create a random batch of classes
            l = torch.randint(low=0, high=2, size=(batch_size, class_num, 8, 4)).float()
            if torch.cuda.is_available():
                x = x.cuda()
                l = l.cuda()
            output0 = focal_loss(x, l)
            output1 = ce(x, l)
            a = float(output0.cpu().detach())
            b = float(output1.cpu().detach())
            if abs(a - b) > max_error:
                max_error = abs(a - b)
        self.assertAlmostEqual(max_error, 0.0, places=3)

    def test_consistency_with_cross_entropy_2d_no_reduction(self):
        """For gamma=0 the focal loss reduces to the cross entropy loss"""
        import numpy as np

        focal_loss = FocalLoss(to_onehot_y=False, gamma=0.0, reduction="none", weight=1.0)
        ce = nn.BCEWithLogitsLoss(reduction="none")
        max_error = 0
        class_num = 10
        batch_size = 128
        for _ in range(100):
            # Create a random tensor of shape (batch_size, class_num, 8, 4)
            x = torch.rand(batch_size, class_num, 8, 4, requires_grad=True)
            # Create a random batch of classes
            l = torch.randint(low=0, high=2, size=(batch_size, class_num, 8, 4)).float()
            if torch.cuda.is_available():
                x = x.cuda()
                l = l.cuda()
            output0 = focal_loss(x, l)
            output1 = ce(x, l)
            a = output0.cpu().detach().numpy()
            b = output1.cpu().detach().numpy()
            error = np.abs(a - b)
            max_error = np.maximum(error, max_error)

        assert np.allclose(max_error, 0)

    def test_consistency_with_cross_entropy_2d_onehot_label(self):
        """For gamma=0 the focal loss reduces to the cross entropy loss"""
        focal_loss = FocalLoss(to_onehot_y=True, gamma=0.0, reduction="mean")
        ce = nn.BCEWithLogitsLoss(reduction="mean")
        max_error = 0
        class_num = 10
        batch_size = 128
        for _ in range(100):
            # Create a random tensor of shape (batch_size, class_num, 8, 4)
            x = torch.rand(batch_size, class_num, 8, 4, requires_grad=True)
            # Create a random batch of classes
            l = torch.randint(low=0, high=class_num, size=(batch_size, 1, 8, 4))
            if torch.cuda.is_available():
                x = x.cuda()
                l = l.cuda()
            output0 = focal_loss(x, l)
            output1 = ce(x, one_hot(l, num_classes=class_num))
            a = float(output0.cpu().detach())
            b = float(output1.cpu().detach())
            if abs(a - b) > max_error:
                max_error = abs(a - b)
        self.assertAlmostEqual(max_error, 0.0, places=3)

    def test_consistency_with_cross_entropy_classification(self):
        """for gamma=0 the focal loss reduces to the cross entropy loss"""
        focal_loss = FocalLoss(to_onehot_y=True, gamma=0.0, reduction="mean")
        ce = nn.BCEWithLogitsLoss(reduction="mean")
        max_error = 0
        class_num = 10
        batch_size = 128
        for _ in range(100):
            # Create a random scores tensor of shape (batch_size, class_num)
            x = torch.rand(batch_size, class_num, requires_grad=True)
            # Create a random batch of classes
            l = torch.randint(low=0, high=class_num, size=(batch_size, 1))
            l = l.long()
            if torch.cuda.is_available():
                x = x.cuda()
                l = l.cuda()
            output0 = focal_loss(x, l)
            output1 = ce(x, one_hot(l, num_classes=class_num))
            a = float(output0.cpu().detach())
            b = float(output1.cpu().detach())
            if abs(a - b) > max_error:
                max_error = abs(a - b)
        self.assertAlmostEqual(max_error, 0.0, places=3)

    def test_consistency_with_cross_entropy_classification_01(self):
        # for gamma=0.1 the focal loss differs from the cross entropy loss
        focal_loss = FocalLoss(to_onehot_y=True, gamma=0.1, reduction="mean")
        ce = nn.BCEWithLogitsLoss(reduction="mean")
        max_error = 0
        class_num = 10
        batch_size = 128
        for _ in range(100):
            # Create a random scores tensor of shape (batch_size, class_num)
            x = torch.rand(batch_size, class_num, requires_grad=True)
            # Create a random batch of classes
            l = torch.randint(low=0, high=class_num, size=(batch_size, 1))
            l = l.long()
            if torch.cuda.is_available():
                x = x.cuda()
                l = l.cuda()
            output0 = focal_loss(x, l)
            output1 = ce(x, one_hot(l, num_classes=class_num))
            a = float(output0.cpu().detach())
            b = float(output1.cpu().detach())
            if abs(a - b) > max_error:
                max_error = abs(a - b)
        self.assertNotAlmostEqual(max_error, 0.0, places=3)

    def test_bin_seg_2d(self):
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 100 * F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float() - 50.0

        # initialize the mean dice loss
        loss = FocalLoss(to_onehot_y=True)

        # focal loss for pred_very_good should be close to 0
        target = target.unsqueeze(1)  # shape (1, 1, H, W)
        focal_loss_good = float(loss(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_empty_class_2d(self):
        num_classes = 2
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float() - 500.0

        # initialize the mean dice loss
        loss = FocalLoss(to_onehot_y=True)

        # focal loss for pred_very_good should be close to 0
        target = target.unsqueeze(1)  # shape (1, 1, H, W)
        focal_loss_good = float(loss(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_multi_class_seg_2d(self):
        num_classes = 6  # labels 0 to 5
        # define 2d examples
        target = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
        # add another dimension corresponding to the batch (batch size = 1 here)
        target = target.unsqueeze(0)  # shape (1, H, W)
        pred_very_good = 1000 * F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float() - 500.0
        # initialize the mean dice loss
        loss = FocalLoss(to_onehot_y=True)
        loss_onehot = FocalLoss(to_onehot_y=False)

        # focal loss for pred_very_good should be close to 0
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)  # test one hot
        target = target.unsqueeze(1)  # shape (1, 1, H, W)

        focal_loss_good = float(loss(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

        focal_loss_good = float(loss_onehot(pred_very_good, target_one_hot).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_bin_seg_3d(self):
        num_classes = 2  # labels 0, 1
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
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3)  # test one hot
        pred_very_good = 1000 * F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float() - 500.0

        # initialize the mean dice loss
        loss = FocalLoss(to_onehot_y=True)
        loss_onehot = FocalLoss(to_onehot_y=False)

        # focal loss for pred_very_good should be close to 0
        target = target.unsqueeze(1)  # shape (1, 1, H, W)
        focal_loss_good = float(loss(pred_very_good, target).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

        focal_loss_good = float(loss_onehot(pred_very_good, target_one_hot).cpu())
        self.assertAlmostEqual(focal_loss_good, 0.0, places=3)

    def test_foreground(self):
        background = torch.ones(1, 1, 5, 5)
        foreground = torch.zeros(1, 1, 5, 5)
        target = torch.cat((background, foreground), dim=1)
        input = torch.cat((background, foreground), dim=1)
        target[:, 0, 2, 2] = 0
        target[:, 1, 2, 2] = 1

        fgbg = FocalLoss(to_onehot_y=False, include_background=True)(input, target)
        fg = FocalLoss(to_onehot_y=False, include_background=False)(input, target)
        self.assertAlmostEqual(float(fgbg.cpu()), 0.1116, places=3)
        self.assertAlmostEqual(float(fg.cpu()), 0.1733, places=3)

    def test_ill_opts(self):
        chn_input = torch.ones((1, 2, 3))
        chn_target = torch.ones((1, 2, 3))
        with self.assertRaisesRegex(ValueError, ""):
            FocalLoss(reduction="unknown")(chn_input, chn_target)

    def test_ill_shape(self):
        chn_input = torch.ones((1, 2, 3))
        chn_target = torch.ones((1, 3))
        with self.assertRaisesRegex(ValueError, ""):
            FocalLoss(reduction="mean")(chn_input, chn_target)

    def test_ill_class_weight(self):
        chn_input = torch.ones((1, 4, 3, 3))
        chn_target = torch.ones((1, 4, 3, 3))
        with self.assertRaisesRegex(ValueError, ""):
            FocalLoss(include_background=True, weight=(1.0, 1.0, 2.0))(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            FocalLoss(include_background=False, weight=(1.0, 1.0, 1.0, 1.0))(chn_input, chn_target)
        with self.assertRaisesRegex(ValueError, ""):
            FocalLoss(include_background=False, weight=(1.0, 1.0, -1.0))(chn_input, chn_target)

    def test_script(self):
        loss = FocalLoss()
        test_input = torch.ones(2, 2, 8, 8)
        test_script_save(loss, test_input, test_input)


if __name__ == "__main__":
    unittest.main()
