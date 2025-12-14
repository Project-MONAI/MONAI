from typing import Union

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    NormalizeIntensity,
)


def get_ct_preprocessing_pipeline():
    """
    CT preprocessing pipeline using standard HU windowing.
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


def get_mri_preprocessing_pipeline():
    """
    MRI preprocessing pipeline using intensity normalization.
    """
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            NormalizeIntensity(nonzero=True),
        ]
    )


def preprocess_dicom_series(
    dicom_path: Union[str, bytes],
    modality: str,
):
    """
    Preprocess a DICOM series based on modality.

    Args:
        dicom_path: Path to DICOM file or directory.
        modality: CT, MR, or MRI.

    Returns:
        Preprocessed image.
    """
    if not isinstance(modality, str):
        raise TypeError("modality must be a string")

    modality = modality.strip().upper()

    if modality == "CT":
        transform = get_ct_preprocessing_pipeline()
    elif modality in ("MR", "MRI"):
        transform = get_mri_preprocessing_pipeline()
    else:
        raise ValueError("Unsupported modality")

    return transform(dicom_path)
