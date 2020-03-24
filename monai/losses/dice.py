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
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `pred` (BNHW[D] where N is number of classes) is compared with ground truth `ground' (BNHW[D]).
    Axis N of `pred` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `ground` can be 1 or N(one-hot format). The `smooth` parameter is a value added to the
    intersection and union components of the inter-over-union calculation to smooth results and prevent divide by 0,
    this value should be small. The `include_background` class attribute can be set to False for an instance of
    DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be background.
    If the non-background segmentations are small compared to the total image size they can get overwhelmed by
    the signal from the background so excluding it in such cases helps convergence.
    """

    def __init__(
        self,
        include_background=True,
        to_onehot_y=False,
        do_sigmoid=False,
        do_softmax=False,
        squared_pred=False,
        jaccard=False
    ):
        """
        Args:
            include_background (bool): If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
            do_sigmoid (bool): If True, apply a sigmoid function to the prediction.
            do_softmax (bool): If True, apply a softmax function to the prediction.
            squared_pred (bool): use squared versions of targets and predictions in the denominator or not.
            jaccard (bool): compute Jaccard Index (soft IoU) instead of dice or not.

        """
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        if do_sigmoid and do_softmax:
            raise ValueError('do_sigmoid=True and do_softmax=Ture are not compatible.')
        self.do_sigmoid = do_sigmoid
        self.do_softmax = do_softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard

    def forward(self, pred, ground, smooth=1e-5):
        """
        Args:
            pred (tensor): the shape should be BNH[WD].
            ground (tensor): the shape should be BNH[WD].
            smooth (float): a small constant to avoid nan.
        """
        if self.do_sigmoid:
            pred = torch.sigmoid(pred)
        n_pred_ch = pred.shape[1]
        if n_pred_ch == 1:
            if self.do_softmax:
                warnings.warn('single channel prediction, `do_softmax=True` ignored.')
            if self.to_onehot_y:
                warnings.warn('single channel prediction, `to_onehot_y=True` ignored.')
            if not self.include_background:
                warnings.warn('single channel prediction, `include_background=False` ignored.')
        else:
            if self.do_softmax:
                pred = torch.softmax(pred, 1)
            if self.to_onehot_y:
                ground = one_hot(ground, n_pred_ch)
            if not self.include_background:
                # if skipping background, removing first channel
                ground = ground[:, 1:]
                pred = pred[:, 1:]
                assert ground.shape == pred.shape, ('ground truth one-hot has differing shape (%r) from pred (%r)' %
                                                    (ground.shape, pred.shape))

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(pred.shape)))
        intersection = torch.sum(ground * pred, reduce_axis)

        if self.squared_pred:
            ground = torch.pow(ground, 2)
            pred = torch.pow(pred, 2)

        ground_o = torch.sum(ground, reduce_axis)
        pred_o = torch.sum(pred, reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator -= intersection

        f = (2.0 * intersection + smooth) / (denominator + smooth)
        return 1.0 - f.mean()  # final reduce_mean across batches and channels


@alias("generalized_dice", "generalised_dice")
class GeneralizedDiceLoss(_Loss):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(
        self,
        include_background=True,
        to_onehot_y=False,
        do_sigmoid=False,
        do_softmax=False,
        w_type='square'
    ):
        """
        Args:
            include_background (bool): If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
            do_sigmoid (bool): If True, apply a sigmoid function to the prediction.
            do_softmax (bool): If True, apply a softmax function to the prediction.
            w_type ('square'|'simple'|'uniform'): type of function to transform ground truth volume to a weight factor.

        """
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        if do_sigmoid and do_softmax:
            raise ValueError('do_sigmoid=True and do_softmax=Ture are not compatible.')
        self.do_sigmoid = do_sigmoid
        self.do_softmax = do_softmax
        self.w_func = torch.ones_like
        if w_type == 'simple':
            self.w_func = torch.reciprocal
        elif w_type == 'square':
            self.w_func = lambda x: torch.reciprocal(x * x)
        else:
            raise ValueError('unknown option for `w_type`: {}'.format(w_type))

    def forward(self, pred, ground, smooth=1e-5):
        """
        Args:
            pred (tensor): the shape should be BNH[WD].
            ground (tensor): the shape should be BNH[WD].
            smooth (float): a small constant to avoid nan.
        """
        if self.do_sigmoid:
            pred = torch.sigmoid(pred)
        n_pred_ch = pred.shape[1]
        if n_pred_ch == 1:
            if self.do_softmax:
                warnings.warn('single channel prediction, `do_softmax=True` ignored.')
            if self.to_onehot_y:
                warnings.warn('single channel prediction, `to_onehot_y=True` ignored.')
            if not self.include_background:
                warnings.warn('single channel prediction, `include_background=False` ignored.')
        else:
            if self.do_softmax:
                pred = torch.softmax(pred, 1)
            if self.to_onehot_y:
                ground = one_hot(ground, n_pred_ch)
            if not self.include_background:
                # if skipping background, removing first channel
                ground = ground[:, 1:]
                pred = pred[:, 1:]
                assert ground.shape == pred.shape, ('ground truth one-hot has differing shape (%r) from pred (%r)' %
                                                    (ground.shape, pred.shape))

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, len(pred.shape)))
        intersection = torch.sum(ground * pred, reduce_axis)

        ground_o = torch.sum(ground, reduce_axis)
        pred_o = torch.sum(pred, reduce_axis)

        denominator = ground_o + pred_o

        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)

        f = (2.0 * intersection * w + smooth) / (denominator * w + smooth)
        return 1.0 - f.mean()  # final reduce_mean across batches and channels
