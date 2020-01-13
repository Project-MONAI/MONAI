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
from torch.nn.modules.loss import _Loss

from monai.networks.utils import one_hot
from monai.utils import export
from monai.utils.aliases import alias


@export("monai.networks.losses")
@alias("dice", "Dice")
class DiceLoss(_Loss):
    """
    Multiclass dice loss. Input logits 'pred' (BNHW[D] where N is number of classes) is compared with ground truth
    `ground' (B1HW[D]). Axis N of `pred' is expected to have logit predictions for each class rather than being image
    channels, while the same axis of `ground' should be 1. If the N channel of `pred' is 1 binary dice loss will be
    calculated. The `smooth' parameter is a value added to the intersection and union components of the inter-over-union
    calculation to smooth results and prevent divide-by-0, this value should be small. The `include_background' class
    attribute can be set to False for an instance of DiceLoss to exclude the first category (channel index 0) which is
    by convention assumed to be background. If the non-background segmentations are small compared to the total image
    size they can get overwhelmed by the signal from the background so excluding it in such cases helps convergence.
    """

    def __init__(self, include_background=True):
        """
        If `include_background` is False channel index 0 (background category) is excluded from the calculation.
        """
        super().__init__()
        self.includeBackground = include_background

    def forward(self, pred, ground, smooth=1e-5):
        if ground.shape[1] != 1:
            raise ValueError("Ground truth should have only a single channel, shape is " + str(ground.shape))

        if pred.shape[1] == 1:  # binary dice loss, use sigmoid activation
            psum = pred.float().sigmoid()
            tsum = ground
        else:
            pinds = (0, 3, 1, 2) if len(ground.shape) == 4 else (0, 4, 1, 2, 3)
            # multiclass dice loss, use softmax in the first dimension and convert target to one-hot encoding
            psum = torch.softmax(pred, 1)
            tsum = one_hot(ground, pred.shape[1])  # BCHW(D) -> BCHW(D)N
            tsum = tsum[:, 0].permute(*pinds).contiguous()  # BCHW(D)N -> BNHW(D)

            assert tsum.shape == pred.shape, ("Ground truth one-hot has differing shape (%r) from source (%r)" %
                                              (tsum.shape, pred.shape))

            # exclude background category so that it doesn't overwhelm the other segmentations if they are small
            if not self.includeBackground:
                tsum = tsum[:, 1:]
                psum = psum[:, 1:]
                pred = pred[:, 1:]

        batchsize = ground.size(0)
        tsum = tsum.float().view(batchsize, -1)
        psum = psum.view(batchsize, -1)

        intersection = psum * tsum
        sums = psum + tsum

        score = 2.0 * (intersection.sum(1) + smooth) / (sums.sum(1) + smooth)
        return 1 - score.sum() / batchsize
