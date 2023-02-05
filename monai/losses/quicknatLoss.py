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
"""
Description
++++++++++++++++++++++
Addition CombinedLosses module which is not part of standard monai loss library.

Usage
++++++++++++++++++++++
Import the package and Instantiate any loss class you want to you::

    from nn_common_modules import losses as additional_losses
    loss = additional_losses.CombinedLoss()

Members
++++++++++++++++++++++
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss


class DiceLoss(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target, weights=None, ignore_index=None, binary=False):
        """
        Forward pass

        :param output: NxCxHxW logits
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :param binary: bool for binarized one chaneel(C=1) input
        :return: torch.tensor
        """
        output = F.softmax(output, dim=1)
        if binary:
            return self._dice_loss_binary(output, target)
        return self._dice_loss_multichannel(output, target, weights, ignore_index)

    @staticmethod
    def _dice_loss_binary(output, target):
        """
        Dice loss for one channel binarized input

        :param output: Nx1xHxW logits
        :param target: NxHxW LongTensor
        :return:
        """
        eps = 0.0001

        intersection = output * target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + target
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        return loss_per_channel.sum() / output.size(1)

    @staticmethod
    def _dice_loss_multichannel(output, target, weights=None, ignore_index=None):
        """
        Forward pass

        :param output: NxCxHxW Variable
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :param binary: bool for binarized one chaneel(C=1) input
        :return:
        """
        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight_mfb=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight_mfb)

    def forward(self, inputs, targets):
        """
        Forward pass

        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        return self.nll_loss(inputs, targets)


class CombinedLoss(_Loss):
    """
    A combination of dice and cross entropy loss
    """

    def __init__(self, weight_mfb=None):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d(weight_mfb)
        self.dice_loss = DiceLoss()
        self.softmax = True
        self.sigmoid = False
        self.to_onehot_y = True
        self.other_act = None

    def forward(self, input, target, weight=None):
        """
        Forward pass
        needs following forms
        input: torch.tensor (NxCxHxW)
        target: torch.tensor (NxHxW)
        weight: torch.tensor (NxHxW)
        to comform with monai standards accept
        :params:input: torch.tensor (NxCxHxW)
        :params:target: torch.tensor (NxCxHxW)
        :params:weight: torch.tensor (NxCxHxW)
        :return: scalar
        """
        # transform of target and weight
        target = target.type(torch.LongTensor)
        target = torch.argmax(target, dim=1)

        if weight is not None:
            weight = weight.type(torch.LongTensor)
            weight = torch.argmax(weight, dim=1)

        input_soft = F.softmax(input, dim=1)

        y_2 = torch.mean(self.dice_loss(input_soft, target))

        if weight is None:
            y_1 = torch.mean(self.cross_entropy_loss.forward(input, target))

        else:
            y_1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))

        return y_1 + y_2
