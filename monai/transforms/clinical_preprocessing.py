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

"""
Clinical preprocessing transforms for medical imaging data.

This module provides modality-specific preprocessing pipelines for common medical imaging modalities.
"""

from typing import Any

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    NormalizeIntensity,
    ScaleIntensityRange,
)


class ModalityTypeError(TypeError):
    """Raised when modality is not a string."""


class UnsupportedModalityError(ValueError):
    """Raised when an unsupported modality is requested."""


def get_ct_preprocessing_pipeline() -> Compose:
    """
    Create a preprocessing pipeline for CT images.

    Returns:
        Compose: Transform composition for CT preprocessing.
            Applies Hounsfield Unit (HU) windowing [-1000, 400] scaled to [0, 1].
            This range captures lung (-1000 to -400 HU) and soft tissue (0 to 100 HU) contrast.
    """
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensityRange(
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )


def get_mri_preprocessing_pipeline() -> Compose:
    """
    Create a preprocessing pipeline for MRI images.

    Returns:
        Compose: Transform composition for MRI preprocessing.
            Normalizes intensities using nonzero voxels only, excluding background regions
            typical in MRI acquisitions.
    """
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            NormalizeIntensity(nonzero=True),
        ]
    )


def preprocess_dicom_series(path: str, modality: str) -> Any:
    """Preprocess a DICOM series or file based on imaging modality.

    Args:
        path: Path to the DICOM file or directory containing a DICOM series.
        modality: Imaging modality. Supported values are "CT", "MR", and "MRI" (case-insensitive).

    Returns:
        Any: Preprocessed image data.

    Raises:
        ModalityTypeError: If modality is not a string.
        UnsupportedModalityError: If the provided modality is not supported.
    """
    if not isinstance(modality, str):
        error_msg = "modality must be a string"
        raise ModalityTypeError(error_msg)

    modality_clean = modality.strip().upper()

    if modality_clean in {"MR", "MRI"}:
        pipeline = get_mri_preprocessing_pipeline()
    elif modality_clean == "CT":
        pipeline = get_ct_preprocessing_pipeline()
    else:
        error_msg = (
            f"Unsupported modality '{modality}'. "
            f"Supported modalities: CT, MR, MRI"
        )
        raise UnsupportedModalityError(error_msg)

    return pipeline(path)


__all__ = [
    "ModalityTypeError",
    "UnsupportedModalityError",
    "get_ct_preprocessing_pipeline",
    "get_mri_preprocessing_pipeline",
    "preprocess_dicom_series",
]