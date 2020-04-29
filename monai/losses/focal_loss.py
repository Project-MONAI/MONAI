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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    PyTorch implementation of the Focal Loss.
    [1] "Focal Loss for Dense Object Detection", T. Lin et al., ICCV 2017
    """
    def __init__(self, gamma=2., alpha=None, reduction='mean'):
        """
        Args:
            gamma: (float) value of the exponent gamma in the definition
            of the Focal loss.
            alpha: (float or float list or None) weights to apply to the
            voxels of each class. If None no weights are applied.
            reduction: (string) Reduction operation to apply on the loss batch.
            It can be 'mean', 'sum' or 'none' as in the standard PyTorch API
            for loss functions.
        """
        super(FocalLoss, self).__init__()
        # same default parameters as in the original paper [1]
        self.gamma = gamma
        self.alpha = alpha  # weight for the classes
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            input: (tensor): the shape should be BNH[WD].
            target: (tensor): the shape should be BNH[WD].
        """
        i = input
        t = target
        # Resize the input and target
        if t.dim() < i.dim():
            # Add a class dimension to the ground-truth segmentation
            t = t.unsqueeze(1)  # N,H,W => N,1,H,W
        if input.dim() > 2:
            i = i.view(i.size(0), i.size(1), -1)  # N,C,H,W => N,C,H*W
            t = t.view(t.size(0), t.size(1), -1)  # N,1,H,W => N,1,H*W
        else:  # Compatibility with classification
            i = i.unsqueeze(2)  # N,C => N,C,1
            t = t.unsqueeze(2)  # N,1 => N,1,1

        # Compute the log proba (more stable numerically than softmax)
        logpt = F.log_softmax(i, dim=1)  # N,C,H*W
        # Keep only log proba values of the ground truth class for each voxel
        logpt = logpt.gather(1, t)  # N,C,H*W => N,1,H*W
        logpt = torch.squeeze(logpt, dim=1)  # N,1,H*W => N,H*W

        # Get the proba
        pt = torch.exp(logpt)  # N,H*W

        if self.alpha is not None:
            if self.alpha.type() != i.data.type():
                self.alpha = self.alpha.type_as(i.data)
            # Select the correct weight for each voxel depending on its
            # associated gt label
            at = torch.unsqueeze(self.alpha, dim=0)  # C => 1,C
            at = torch.unsqueeze(at, dim=2)  # 1,C => 1,C,1
            at = at.expand((t.size(0), -1, t.size(2)))  # 1,C,1 => N,C,H*W
            at = at.gather(1, t.data)  # selection of the weights  => N,1,H*W
            at = torch.squeeze(at, dim=1)  # N,1,H*W => N,H*W
            # Multiply the log proba by their weights
            logpt = logpt * Variable(at)

        # Compute the loss mini-batch
        weight = torch.pow(-pt + 1., self.gamma)
        loss = torch.mean(-weight * logpt, dim=1)  # N

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        # Default is mean reduction
        else:
            return loss.mean()
