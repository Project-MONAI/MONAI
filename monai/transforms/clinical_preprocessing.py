import pytest
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensityRange, NormalizeIntensity
from monai.transforms.clinical_preprocessing import (
    get_ct_preprocessing_pipeline,
    get_mri_preprocessing_pipeline,
    preprocess_dicom_series,
    UnsupportedModalityError,
    ModalityTypeError,
)


def test_ct_preprocessing_pipeline():
    """Test CT preprocessing pipeline returns expected transform composition and parameters."""
    pipeline = get_ct_preprocessing_pipeline()
    assert hasattr(pipeline, 'transforms')
    assert len(pipeline.transforms) == 3
    assert isinstance(pipeline.transforms[0], LoadImage)
    assert isinstance(pipeline.transforms[1], EnsureChannelFirst)
    assert isinstance(pipeline.transforms[2], ScaleIntensityRange)

    # Verify CT-specific HU window parameters
    scale_transform = pipeline.transforms[2]
    assert scale_transform.a_min == -1000
    assert scale_transform.a_max == 400
    assert scale_transform.b_min == 0.0
    assert scale_transform.b_max == 1.0
    assert scale_transform.clip is True


def test_mri_preprocessing_pipeline():
    """Test MRI preprocessing pipeline returns expected transform composition and parameters."""
    pipeline = get_mri_preprocessing_pipeline()
    assert hasattr(pipeline, 'transforms')
    assert len(pipeline.transforms) == 3
    assert isinstance(pipeline.transforms[0], LoadImage)
    assert isinstance(pipeline.transforms[1], EnsureChannelFirst)
    assert isinstance(pipeline.transforms[2], NormalizeIntensity)

    # Verify MRI-specific normalization parameter
    normalize_transform = pipeline.transforms[2]
    assert normalize_transform.nonzero is True


def test_preprocess_dicom_series_invalid_modality():
    """Test preprocess_dicom_series raises UnsupportedModalityError for unsupported modality."""
    with pytest.raises(UnsupportedModalityError, match=r"Unsupported modality.*PET.*CT, MR, MRI"):
        preprocess_dicom_series("dummy_path.dcm", "PET")


def test_preprocess_dicom_series_invalid_type():
    """Test preprocess_dicom_series raises ModalityTypeError for non-string modality."""
    with pytest.raises(ModalityTypeError, match=r"modality must be a string, got int"):
        preprocess_dicom_series("dummy_path.dcm", 123)
