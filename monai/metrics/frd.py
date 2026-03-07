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

from monai.metrics.fid import get_fid_score
from monai.metrics.metric import Metric

__all__ = ["FrechetRadiomicsDistance", "get_frd_score"]


class FrechetRadiomicsDistance(Metric):
    """
    Fréchet Radiomics Distance (FRD). Computes the Fréchet distance between two
    distributions of radiomic feature vectors, in the same way as the Fréchet
    Inception Distance (FID) but for radiomics-based features.

    Unlike FID, FRD uses interpretable, clinically relevant radiomic features
    (e.g. from PyRadiomics) and works for both 2D and 3D images, with optional
    conditioning by anatomical masks. See Konz et al. "Fréchet Radiomic Distance
    (FRD): A Versatile Metric for Comparing Medical Imaging Datasets."
    https://arxiv.org/abs/2412.01496

    This metric accepts two groups of pre-extracted radiomic feature vectors with
    shape (number of samples, number of features). The same Fréchet distance
    formula as in FID is applied to the mean and covariance of these features.

    Args:
        y_pred: Radiomic feature vectors for the first distribution (e.g. from
            generated or reconstructed images), shape (N, F).
        y: Radiomic feature vectors for the second distribution (e.g. from real
            images), shape (N, F).

    Returns:
        Scalar tensor containing the FRD value.
    """

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return get_frd_score(y_pred, y)


def get_frd_score(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the FRD score from two batches of radiomic feature vectors.

    The implementation reuses the same Fréchet distance as FID; only the
    semantics (radiomic features vs. deep features) differ.

    Args:
        y_pred: Feature vectors for the first distribution, shape (N, F).
        y: Feature vectors for the second distribution, shape (N, F).

    Returns:
        Scalar tensor containing the Fréchet Radiomics Distance.
    """
    return get_fid_score(y_pred, y)
