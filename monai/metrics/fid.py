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

import numpy as np
import torch

from monai.metrics.metric import Metric
from monai.utils import optional_import

scipy, _ = optional_import("scipy")


class FIDMetric(Metric):
    """
    Frechet Inception Distance (FID). The FID calculates the distance between two distributions of feature vectors.
    Based on: Heusel M. et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium."
    https://arxiv.org/abs/1706.08500. The inputs for this metric should be two groups of feature vectors (with format
    (number images, number of features)) extracted from a pretrained network.

    Originally, it was proposed to use the activations of the pool_3 layer of an Inception v3 pretrained with Imagenet.
    However, others networks pretrained on medical datasets can be used as well (for example, RadImageNwt for 2D and
    MedicalNet for 3D images). If the chosen model output is not a scalar, a global spatia average pooling should be
    used.
    """

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return get_fid_score(y_pred, y)


def get_fid_score(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the FID score metric on a batch of feature vectors.

    Args:
        y_pred: feature vectors extracted from a pretrained network run on generated images.
        y: feature vectors extracted from a pretrained network run on images from the real data distribution.
    """
    y = y.double()
    y_pred = y_pred.double()

    if y.ndimension() > 2:
        raise ValueError("Inputs should have (number images, number of features) shape.")

    mu_y_pred = torch.mean(y_pred, dim=0)
    sigma_y_pred = _cov(y_pred, rowvar=False)
    mu_y = torch.mean(y, dim=0)
    sigma_y = _cov(y, rowvar=False)

    return compute_frechet_distance(mu_y_pred, sigma_y_pred, mu_y, sigma_y)


def _cov(input_data: torch.Tensor, rowvar: bool = True) -> torch.Tensor:
    """
    Estimate a covariance matrix of the variables.

    Args:
        input_data: A 1-D or 2-D array containing multiple variables and observations. Each row of `m` represents a variable,
            and each column a single observation of all those variables.
        rowvar: If rowvar is True (default), then each row represents a variable, with observations in the columns.
            Otherwise, the relationship is transposed: each column represents a variable, while the rows contain
            observations.
    """
    if input_data.dim() < 2:
        input_data = input_data.view(1, -1)

    if not rowvar and input_data.size(0) != 1:
        input_data = input_data.t()

    factor = 1.0 / (input_data.size(1) - 1)
    input_data = input_data - torch.mean(input_data, dim=1, keepdim=True)
    return factor * input_data.matmul(input_data.t()).squeeze()


def _sqrtm(input_data: torch.Tensor) -> torch.Tensor:
    """Compute the square root of a matrix."""
    scipy_res, _ = scipy.linalg.sqrtm(input_data.detach().cpu().numpy().astype(np.float64), disp=False)
    return torch.from_numpy(scipy_res)


def compute_frechet_distance(
    mu_x: torch.Tensor, sigma_x: torch.Tensor, mu_y: torch.Tensor, sigma_y: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """The Frechet distance between multivariate normal distributions."""
    diff = mu_x - mu_y

    covmean = _sqrtm(sigma_x.mm(sigma_y))

    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        print(f"FID calculation produces singular product; adding {epsilon} to diagonal of covariance estimates")
        offset = torch.eye(sigma_x.size(0), device=mu_x.device, dtype=mu_x.dtype) * epsilon
        covmean = _sqrtm((sigma_x + offset).mm(sigma_y + offset))

    # Numerical error might give slight imaginary component
    if torch.is_complex(covmean):
        if not torch.allclose(torch.diagonal(covmean).imag, torch.tensor(0, dtype=torch.double), atol=1e-3):
            raise ValueError(f"Imaginary component {torch.max(torch.abs(covmean.imag))} too high.")
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * tr_covmean
