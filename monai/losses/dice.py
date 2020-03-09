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

import warnings

import torch
from torch.nn.modules.loss import _Loss

from monai.networks.utils import one_hot
from monai.utils import export
from monai.utils.aliases import alias


@export("monai.losses")
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
        if do_sigmoid and do_softmax:
            raise ValueError('do_sigmoid=True and do_softmax=Ture are not compatible.')
        self.do_sigmoid = do_sigmoid
        self.do_softmax = do_softmax

    def forward(self, pred, ground, smooth=1e-5):
        if ground.shape[1] != 1:
            raise ValueError("Ground truth should have only a single channel, shape is " + str(ground.shape))

        psum = pred.float()
        if self.do_sigmoid:
            psum = psum.sigmoid()  # use sigmoid activation
        if pred.shape[1] == 1:
            if self.do_softmax:
                raise ValueError('do_softmax is not compatible with single channel prediction.')
            if not self.include_background:
                warnings.warn('single channel prediction, `include_background=False` ignored.')
            tsum = ground
        else:  # multiclass dice loss
            if self.do_softmax:
                psum = torch.softmax(pred, 1)
            tsum = one_hot(ground, pred.shape[1])  # B1HW(D) -> BNHW(D)
            # exclude background category so that it doesn't overwhelm the other segmentations if they are small
            if not self.include_background:
                tsum = tsum[:, 1:]
                psum = psum[:, 1:]
        assert tsum.shape == psum.shape, ("Ground truth one-hot has differing shape (%r) from source (%r)" %
                                          (tsum.shape, psum.shape))

        batchsize, n_classes = tsum.shape[:2]
        tsum = tsum.float().view(batchsize, n_classes, -1)
        psum = psum.view(batchsize, n_classes, -1)

        intersection = psum * tsum
        sums = psum + tsum

        score = (2.0 * intersection.sum(2) + smooth) / (sums.sum(2) + smooth)
        return 1 - score.mean()


@alias("generalized_dice", "generalised_dice")
class GeneralizedDiceLoss(_Loss):
    """
    Compute the generalised Dice loss defined in:
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(self, include_background=True, do_sigmoid=False, do_softmax=False, w_type='square'):
        """
        Args:
            include_background (bool): If False channel index 0 (background category) is excluded from the calculation.
            do_sigmoid (bool): If True, apply a sigmoid function to the prediction.
            do_softmax (bool): If True, apply a softmax function to the prediction.
            w_type ('square'|'simple'|'uniform'): type of function to transform ground truth volume to a weight factor.
        """
        super().__init__()
        self.include_background = include_background
        if do_sigmoid and do_softmax:
            raise ValueError('do_sigmoid=True and do_softmax=Ture are not compatible.')
        self.do_sigmoid = do_sigmoid
        self.do_softmax = do_softmax

        self.w_func = torch.ones_like
        if w_type == 'simple':
            self.w_func = lambda x: torch.reciprocal(x)
        elif w_type == 'square':
            self.w_func = lambda x: torch.reciprocal(x * x)
        else:
            raise ValueError('unknown option for `w_type`: {}'.format(w_type))

    def forward(self, pred, ground, smooth=1e-5):
        """
        Args:
            pred (tensor): the shape should be BNH[WD].
            ground (tensor): the shape should be B1H[WD].
            smooth (float): a small constant to avoid nan.
        """
        if ground.shape[1] != 1:
            raise ValueError("Ground truth should have only a single channel, shape is " + str(ground.shape))

        psum = pred.float()
        if self.do_sigmoid:
            psum = psum.sigmoid()  # use sigmoid activation
        if pred.shape[1] == 1:
            if self.do_softmax:
                raise ValueError('do_softmax is not compatible with single channel prediction.')
            if not self.include_background:
                warnings.warn('single channel prediction, `include_background=False` ignored.')
            tsum = ground
        else:  # multiclass dice loss
            if self.do_softmax:
                psum = torch.softmax(pred, 1)
            tsum = one_hot(ground, pred.shape[1])  # B1HW(D) -> BNHW(D)
            # exclude background category so that it doesn't overwhelm the other segmentations if they are small
            if not self.include_background:
                tsum = tsum[:, 1:]
                psum = psum[:, 1:]
        assert tsum.shape == psum.shape, ("Ground truth one-hot has differing shape (%r) from source (%r)" %
                                          (tsum.shape, psum.shape))

        batchsize, n_classes = tsum.shape[:2]
        tsum = tsum.float().view(batchsize, n_classes, -1)
        psum = psum.view(batchsize, n_classes, -1)

        intersection = psum * tsum
        sums = psum + tsum

        w = self.w_func(tsum.sum(2))
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        score = (2.0 * intersection.sum(2) * w + smooth) / (sums.sum(2) * w + smooth)
        return 1 - score.mean()
