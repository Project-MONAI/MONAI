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

from collections.abc import Callable

import torch

from monai.metrics.metric import Metric


class MMDMetric(Metric):
    """
    Unbiased Maximum Mean Discrepancy (MMD) is a kernel-based method for measuring the similarity between two
    distributions. It is a non-negative metric where a smaller value indicates a closer match between the two
    distributions.

    Gretton, A., et al,, 2012.  A kernel two-sample test. The Journal of Machine Learning Research, 13(1), pp.723-773.

    Args:
        y_mapping: Callable to transform the y tensors before computing the metric. It is usually a Gaussian or Laplace
            filter, but it can be any function that takes a tensor as input and returns a tensor as output such as a
            feature extractor or an Identity function., e.g. `y_mapping = lambda x: x.square()`.
    """

    def __init__(self, y_mapping: Callable | None = None) -> None:
        super().__init__()
        self.y_mapping = y_mapping

    def __call__(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return compute_mmd(y, y_pred, self.y_mapping)


def compute_mmd(y: torch.Tensor, y_pred: torch.Tensor, y_mapping: Callable | None) -> torch.Tensor:
    """
    Args:
        y: first sample (e.g., the reference image). Its shape is (B,C,W,H) for 2D data and (B,C,W,H,D) for 3D.
        y_pred: second sample (e.g., the reconstructed image). It has similar shape as y.
        y_mapping: Callable to transform the y tensors before computing the metric.
    """
    if y_pred.shape[0] == 1 or y.shape[0] == 1:
        raise ValueError("MMD metric requires at least two samples in y and y_pred.")

    if y_mapping is not None:
        y = y_mapping(y)
        y_pred = y_mapping(y_pred)

    if y_pred.shape != y.shape:
        raise ValueError(
            "y_pred and y shapes dont match after being processed "
            f"by their transforms, received y_pred: {y_pred.shape} and y: {y.shape}"
        )

    for d in range(len(y.shape) - 1, 1, -1):
        y = y.squeeze(dim=d)
        y_pred = y_pred.squeeze(dim=d)

    y = y.view(y.shape[0], -1)
    y_pred = y_pred.view(y_pred.shape[0], -1)

    y_y = torch.mm(y, y.t())
    y_pred_y_pred = torch.mm(y_pred, y_pred.t())
    y_pred_y = torch.mm(y_pred, y.t())

    m = y.shape[0]
    n = y_pred.shape[0]

    # Ref. 1 Eq. 3 (found under Lemma 6)
    # term 1
    c1 = 1 / (m * (m - 1))
    a = torch.sum(y_y - torch.diag(torch.diagonal(y_y)))

    # term 2
    c2 = 1 / (n * (n - 1))
    b = torch.sum(y_pred_y_pred - torch.diag(torch.diagonal(y_pred_y_pred)))

    # term 3
    c3 = 2 / (m * n)
    c = torch.sum(y_pred_y)

    mmd = c1 * a + c2 * b - c3 * c
    return mmd
