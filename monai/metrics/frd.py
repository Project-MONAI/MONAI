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
    Inception Distance (FID) but applied to radiomics-based features instead of
    deep-network embeddings.

    Unlike FID, FRD uses interpretable, clinically relevant radiomic features
    (e.g. extracted via PyRadiomics), which makes it directly applicable to both
    2D and 3D images and allows optional conditioning by anatomical masks —
    all handled during upstream feature extraction, not by this class. See
    Konz et al. "Fréchet Radiomic Distance (FRD): A Versatile Metric for
    Comparing Medical Imaging Datasets." https://arxiv.org/abs/2412.01496

    This class accepts pre-extracted radiomic feature tensors of shape (N, F)
    and applies the same Fréchet distance formula as FID to the empirical means
    and covariances of those features.
    """

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute FRD between two sets of pre-extracted radiomic feature vectors.

        Args:
            y_pred: Radiomic feature vectors for the first distribution (e.g. from
                generated or reconstructed images), shape (N, F) with N >= 2.
            y: Radiomic feature vectors for the second distribution (e.g. from real
                images), shape (N, F) with N >= 2.

        Returns:
            Scalar tensor containing the FRD value.

        Raises:
            ValueError: When either tensor is not exactly 2-dimensional or has
                fewer than 2 samples.
        """
        return get_frd_score(y_pred, y)


def get_frd_score(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the FRD score from two batches of radiomic feature vectors.

    The implementation reuses the same Fréchet distance as FID; only the
    semantics (radiomic features vs. deep network features) differ.

    Args:
        y_pred: Feature vectors for the first distribution, shape (N, F) with N >= 2.
        y: Feature vectors for the second distribution, shape (N, F) with N >= 2.

    Returns:
        Scalar tensor containing the Fréchet Radiomics Distance.

    Raises:
        ValueError: When either tensor is not exactly 2-dimensional (i.e. not
            shape (N, F)), or when either tensor has fewer than 2 samples
            (required for covariance estimation).
    """
    for name, t in (("y_pred", y_pred), ("y", y)):
        if t.ndimension() != 2:
            raise ValueError(
                f"{name} must be a 2-D tensor of shape (N, F) — got shape {tuple(t.shape)}. "
                "Pass pre-extracted radiomic feature vectors, not raw images."
            )
        if t.size(0) < 2:
            raise ValueError(
                f"{name} must contain at least 2 samples for covariance estimation — got {t.size(0)}."
            )
    return get_fid_score(y_pred, y)
