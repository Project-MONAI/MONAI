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

SUPPORTED_MODALITIES = ("CT", "MR", "MRI")


class UnsupportedModalityError(ValueError):
    """Raised when an unsupported modality is provided."""
    pass


class ModalityTypeError(TypeError):
    """Raised when modality is not a string."""
    pass


def get_ct_preprocessing_pipeline() -> Compose:
    """Return a CT preprocessing pipeline."""
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    ])


def get_mri_preprocessing_pipeline() -> Compose:
    """Return an MRI preprocessing pipeline."""
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        NormalizeIntensity(nonzero=True),
    ])


def preprocess_dicom_series(
    dicom_path: Union[str, bytes, PathLike],
    modality: str,
) -> MetaTensor:
    """Preprocess a DICOM series according to modality (CT or MRI).

    Args:
        dicom_path (Union[str, bytes, PathLike]): Path to DICOM series.
        modality (str): Modality type, must be one of 'CT', 'MR', 'MRI'.

    Returns:
        MetaTensor: Preprocessed image tensor.

    Raises:
        ModalityTypeError: If modality is not a string.
        UnsupportedModalityError: If modality is not supported.
    """
    if not isinstance(modality, str):
        raise ModalityTypeError(f"modality must be a string, got {type(modality).__name__}")

    modality = modality.strip().upper()

    if modality == "CT":
        transform = get_ct_preprocessing_pipeline()
    elif modality in ("MR", "MRI"):
        transform = get_mri_preprocessing_pipeline()
    else:
        raise UnsupportedModalityError(
            f"Unsupported modality: {modality}. Supported values: {', '.join(SUPPORTED_MODALITIES)}"
        )

    return transform(dicom_path)
