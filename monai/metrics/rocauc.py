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

from sklearn.metrics import roc_auc_score
from monai.networks.utils import one_hot


def compute_roc_auc(y_pred, y, to_onehot_y=False, add_softmax=False, add_sigmoid=False):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) based on:
    `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ .

    Args:
        y_pred (torch.Tensor): input data to compute, typical classification model output.
            it must be One-Hot format and first dim is batch, example shape: [16] or [16, 2].
        y (torch.Tensor): ground truth to compute ROC AUC metric, the first dim is batch.
            example shape: [16, 1] will be converted into [16, 3].
        to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
        add_softmax (bool): whether to add softmax function to y_pred before computation. Defaults to False.
        add_sigmoid (bool): whether to add sigmoid function to y_pred before computation. Defaults to False.

    Note:
        ROC_AUC expects y to be comprised of 0's and 1's.
        y_pred must either be probability estimates or confidence values.

    """
    if add_softmax and add_sigmoid:
        raise ValueError('add_softmax=True and add_sigmoid=True are not compatible.')
    y_pred_ndim = y_pred.ndimension()
    if y_pred_ndim not in (1, 2):
        raise ValueError("predictions should be of shape (batch_size, n_classes) or (batch_size, ).")
    if y.ndimension() not in (1, 2):
        raise ValueError("targets should be of shape (batch_size, n_classes) or (batch_size, ).")

    n_classes = y_pred.shape[1] if y_pred_ndim == 2 else 1
    if n_classes == 1:
        if to_onehot_y:
            warnings.warn('y_pred has only one channel, to_onehot_y=True ignored.')
        if add_softmax:
            warnings.warn('y_pred has only one channel, add_softmax=True ignored.')
    else:
        if to_onehot_y:
            y = one_hot(y, n_classes)
        if add_softmax:
            y_pred = y_pred.float().softmax(dim=1)
    if add_sigmoid:
        y_pred = y_pred.float().sigmoid()

    return roc_auc_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
