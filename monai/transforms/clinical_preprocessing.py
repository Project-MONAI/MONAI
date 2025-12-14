"""Clinical DICOM preprocessing utilities for CT and MRI modalities."""

from typing import Union
from os import PathLike

from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    NormalizeIntensity,
)

# Use a tuple for programmatic checks and formatting
SUPPORTED_MODALITIES = ("CT", "MR", "MRI")


def get_ct_preprocessing_pipeline() -> Compose:
    """
    Build a CT preprocessing pipeline using standard HU windowing.

    The pipeline applies LoadImage, EnsureChannelFirst, and ScaleIntensityRange
    with HU window [-1000, 400] normalized to [0.0, 1.0].

    Returns:
        Compose: A composed transform pipeline for CT preprocessing.
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
    Build an MRI preprocessing pipeline using intensity normalization.

    The pipeline applies LoadImage, EnsureChannelFirst, and NormalizeIntensity
    with nonzero=True to normalize only non-zero voxels.

    Returns:
        Compose: A composed transform pipeline for MRI preprocessing.
    """
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            NormalizeIntensity(nonzero=True),
        ]
    )


def preprocess_dicom_series(
    dicom_path: Union[str, bytes, PathLike],
    modality: str,
) -> MetaTensor:
    """
    Preprocess a DICOM series based on modality.

    Args:
        dicom_path: Path to DICOM file or directory.
        modality: Imaging modality. Supported values: "CT", "MR", "MRI" (case-insensitive).

    Returns:
        MetaTensor: Preprocessed image with intensity values normalized based on modality.

    Raises:
        TypeError: If modality is not a string.
        ValueError: If modality is not one of the supported values.
    """
    if not isinstance(modality, str):
        raise TypeError(f"modality must be a string, got {type(modality).__name__}")

    modality = modality.strip().upper()

    if modality == "CT":
        transform = get_ct_preprocessing_pipeline()
    elif modality in ("MR", "MRI"):
        transform = get_mri_preprocessing_pipeline()
    else:
        raise ValueError(
            f"Unsupported modality: {modality}. Supported values: {', '.join(SUPPORTED_MODALITIES)}"
        )

    return transform(dicom_path)
