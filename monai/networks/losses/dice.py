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

    def __init__(self, include_background=True, do_sigmoid=False, do_softmax=False):
        """
        Args:
            include_background (bool): If False channel index 0 (background category) is excluded from the calculation.
            do_sigmoid (bool): If True, apply a sigmoid function to the prediction.
            do_softmax (bool): If True, apply a softmax function to the prediction.
        """
        super().__init__()
        self.include_background = include_background
        self.do_sigmoid = do_sigmoid
        self.do_softmax = do_softmax

    def forward(self, pred, ground, smooth=1e-5):
        if ground.shape[1] != 1:
            raise ValueError("Ground truth should have only a single channel, shape is " + str(ground.shape))

        psum = pred.float()
        if self.do_sigmoid:
            psum = psum.sigmoid()
        if pred.shape[1] == 1:  # binary dice loss, use sigmoid activation
            if self.do_softmax:
                raise ValueError('do_softmax is not compatible with single channel prediction.')
            if not self.include_background:
                raise RuntimeWarning('single channel ground truth, `include_background=False` ignored.')
            tsum = ground
        else:
            if self.do_softmax:
                if self.do_sigmoid:
                    raise ValueError('do_sigmoid=True and do_softmax=Ture are not compatible.')
                # multiclass dice loss, use softmax in the first dimension and convert target to one-hot encoding
                psum = torch.softmax(pred, 1)
            tsum = one_hot(ground, pred.shape[1])  # B1HW(D) -> BNHW(D)
            # exclude background category so that it doesn't overwhelm the other segmentations if they are small
            if not self.include_background:
                tsum = tsum[:, 1:]
                psum = psum[:, 1:]
        assert tsum.shape == pred.shape, ("Ground truth one-hot has differing shape (%r) from source (%r)" %
                                          (tsum.shape, pred.shape))

        batchsize = ground.size(0)
        tsum = tsum.float().view(batchsize, -1)
        psum = psum.view(batchsize, -1)

        intersection = psum * tsum
        sums = psum + tsum

        score = 2.0 * (intersection.sum(1) + smooth) / (sums.sum(1) + smooth)
        return 1 - score.sum() / batchsize
