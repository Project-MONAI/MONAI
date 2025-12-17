"""
Clinical preprocessing transforms for medical imaging data.

This module provides preprocessing pipelines for different medical imaging modalities.
"""

from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensityRange, NormalizeIntensity
from monai.data import MetaTensor


class ModalityTypeError(TypeError):
    """Exception raised when modality parameter is not a string."""
    pass


class UnsupportedModalityError(ValueError):
    """Exception raised when an unsupported modality is requested."""
    pass


def get_ct_preprocessing_pipeline() -> Compose:
    """
    Create a preprocessing pipeline for CT (Computed Tomography) images.
    
    Returns:
        Compose: A transform composition for CT preprocessing.
        
    The pipeline consists of:
    1. LoadImage - Load DICOM series
    2. EnsureChannelFirst - Add channel dimension
    3. ScaleIntensityRange - Scale Hounsfield Units (HU) from [-1000, 400] to [0, 1]
    
    Note:
        The HU window [-1000, 400] is a common soft tissue window.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True)
    ])


def get_mri_preprocessing_pipeline() -> Compose:
    """
    Create a preprocessing pipeline for MRI (Magnetic Resonance Imaging) images.
    
    Returns:
        Compose: A transform composition for MRI preprocessing.
        
    The pipeline consists of:
    1. LoadImage - Load DICOM series
    2. EnsureChannelFirst - Add channel dimension
    3. NormalizeIntensity - Normalize non-zero voxels
    
    Note:
        Normalization is applied only to non-zero voxels to avoid bias from background.
    """
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        NormalizeIntensity(nonzero=True)
    ])


def preprocess_dicom_series(path: str, modality: str) -> MetaTensor:
    """
    Preprocess a DICOM series based on the imaging modality.
    
    Args:
        path: Path to the DICOM series directory or file.
        modality: Imaging modality (case-insensitive). Supported values:
                  "CT", "MR", "MRI" (MRI is treated as synonym for MR).
    
    Returns:
        MetaTensor: The preprocessed image data with metadata.
    
    Raises:
        ModalityTypeError: If modality is not a string.
        UnsupportedModalityError: If modality is not supported.
    """
    # Validate input type
    if not isinstance(modality, str):
        raise ModalityTypeError(f"modality must be a string, got {type(modality).__name__}")
    
    # Normalize modality string (strip whitespace, convert to uppercase)
    modality_clean = modality.strip().upper()
    
    # Map MRI to MR (treat as synonyms)
    if modality_clean == "MRI":
        modality_clean = "MR"
    
    # Select appropriate preprocessing pipeline
    if modality_clean == "CT":
        pipeline = get_ct_preprocessing_pipeline()
    elif modality_clean == "MR":
        pipeline = get_mri_preprocessing_pipeline()
    else:
        supported = ["CT", "MR", "MRI"]
        raise UnsupportedModalityError(
            f"Unsupported modality '{modality}'. Supported modalities: {', '.join(supported)}"
        )
    
    # Apply preprocessing pipeline
    return pipeline(path)


# Export the public API
__all__ = [
    "ModalityTypeError",
    "UnsupportedModalityError", 
    "get_ct_preprocessing_pipeline",
    "get_mri_preprocessing_pipeline",
    "preprocess_dicom_series",
]