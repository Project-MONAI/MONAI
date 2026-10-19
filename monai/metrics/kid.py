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

import torch

from monai.metrics.metric import Metric


class KIDMetric(Metric):
    """
    Kernel Inception Distance (KID). The KID is the squared Maximum Mean Discrepancy (MMD) between two distributions
    of feature vectors, computed with a polynomial kernel ``k(x, y) = (gamma * x.y + coef0) ** degree``.
    Based on: Binkowski M. et al. "Demystifying MMD GANs." https://arxiv.org/abs/1801.01401.

    Unlike FID, KID has an unbiased estimator, which makes it more reliable for the small sample sizes common in
    medical imaging. Because the estimator is unbiased, it can return small negative values when the two
    distributions are very close.

    The inputs for this metric should be two groups of feature vectors (with format (number images, number of
    features)) extracted from a pretrained network. Originally, it was proposed to use the activations of the pool_3
    layer of an Inception v3 pretrained with Imagenet. However, other networks pretrained on medical datasets can be
    used as well (for example, RadImageNet for 2D and MedicalNet for 3D images). If the chosen model output is not a
    vector, a global spatial average pooling should be used.

    Args:
        subset_size: if given, the score is averaged over ``num_subsets`` random subsets of this many samples drawn
            without replacement from each input, as in the original paper. If ``None`` (default), all samples are used
            in a single estimate.
        num_subsets: number of random subsets to average over. Only used when ``subset_size`` is given.
        degree: degree of the polynomial kernel.
        gamma: scale of the dot product in the polynomial kernel. If ``None`` (default), ``1 / number of features``.
        coef0: constant term of the polynomial kernel.
        generator: optional random number generator used to draw the subsets, for reproducibility.
    """

    def __init__(
        self,
        subset_size: int | None = None,
        num_subsets: int = 100,
        degree: int = 3,
        gamma: float | None = None,
        coef0: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if subset_size is not None and subset_size < 2:
            raise ValueError(f"subset_size should be at least 2, got {subset_size}.")
        if num_subsets < 1:
            raise ValueError(f"num_subsets should be at least 1, got {num_subsets}.")
        self.subset_size = subset_size
        self.num_subsets = num_subsets
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.generator = generator

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return get_kid_score(
            y_pred,
            y,
            subset_size=self.subset_size,
            num_subsets=self.num_subsets,
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            generator=self.generator,
        )


def get_kid_score(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    subset_size: int | None = None,
    num_subsets: int = 100,
    degree: int = 3,
    gamma: float | None = None,
    coef0: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Computes the KID score metric on a batch of feature vectors.

    Args:
        y_pred: feature vectors extracted from a pretrained network run on generated images.
        y: feature vectors extracted from a pretrained network run on images from the real data distribution.
        subset_size: if given, average the score over ``num_subsets`` random subsets of this many samples.
        num_subsets: number of random subsets to average over. Only used when ``subset_size`` is given.
        degree: degree of the polynomial kernel.
        gamma: scale of the dot product in the polynomial kernel. If ``None``, ``1 / number of features``.
        coef0: constant term of the polynomial kernel.
        generator: optional random number generator used to draw the subsets.
    """
    y = y.double()
    y_pred = y_pred.double()

    if y.ndimension() != 2 or y_pred.ndimension() != 2:
        raise ValueError("Inputs should have (number images, number of features) shape.")
    if y.shape[1] != y_pred.shape[1]:
        raise ValueError(
            f"Inputs should have the same number of features, received y_pred: {y_pred.shape} and y: {y.shape}."
        )
    if y.shape[0] < 2 or y_pred.shape[0] < 2:
        raise ValueError("KID metric requires at least two samples in y and y_pred.")

    if subset_size is None:
        return compute_kernel_distance(y_pred, y, degree=degree, gamma=gamma, coef0=coef0)

    if subset_size > min(y.shape[0], y_pred.shape[0]):
        raise ValueError(
            f"subset_size ({subset_size}) should not be larger than the number of samples, "
            f"received y_pred: {y_pred.shape[0]} and y: {y.shape[0]}."
        )

    scores = []
    for _ in range(num_subsets):
        idx_pred = torch.randperm(y_pred.shape[0], generator=generator)[:subset_size].to(y_pred.device)
        idx = torch.randperm(y.shape[0], generator=generator)[:subset_size].to(y.device)
        scores.append(compute_kernel_distance(y_pred[idx_pred], y[idx], degree=degree, gamma=gamma, coef0=coef0))
    return torch.stack(scores).mean()


def _polynomial_kernel(x: torch.Tensor, y: torch.Tensor, degree: int, gamma: float, coef0: float) -> torch.Tensor:
    """Polynomial kernel ``(gamma * x.y + coef0) ** degree`` between all pairs of rows of x and y."""
    return (gamma * x.mm(y.t()) + coef0) ** degree


def compute_kernel_distance(
    x: torch.Tensor, y: torch.Tensor, degree: int = 3, gamma: float | None = None, coef0: float = 1.0
) -> torch.Tensor:
    """
    The unbiased estimator of the squared Maximum Mean Discrepancy between two sets of feature vectors with a
    polynomial kernel (Binkowski et al. 2018, Eq. 4; Gretton et al. 2012, Lemma 6). The two sets may have a different
    number of samples.

    Args:
        x: first set of feature vectors, with shape (m, number of features).
        y: second set of feature vectors, with shape (n, number of features).
        degree: degree of the polynomial kernel.
        gamma: scale of the dot product in the polynomial kernel. If ``None``, ``1 / number of features``.
        coef0: constant term of the polynomial kernel.
    """
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    m = x.shape[0]
    n = y.shape[0]

    k_xx = _polynomial_kernel(x, x, degree, gamma, coef0)
    k_yy = _polynomial_kernel(y, y, degree, gamma, coef0)
    k_xy = _polynomial_kernel(x, y, degree, gamma, coef0)

    # the diagonal terms k(x_i, x_i) are excluded to make the estimator unbiased
    term_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (m * (m - 1))
    term_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1))
    term_xy = k_xy.sum() / (m * n)

    return term_xx + term_yy - 2 * term_xy
