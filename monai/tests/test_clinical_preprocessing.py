import pytest
from unittest.mock import patch, Mock
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

    # Verify LoadImage configuration (as suggested in review)
    load_transform = pipeline.transforms[0]
    assert load_transform.image_only is True


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

    # Verify LoadImage configuration (as suggested in review)
    load_transform = pipeline.transforms[0]
    assert load_transform.image_only is True


def test_preprocess_dicom_series_invalid_modality():
    """Test preprocess_dicom_series raises UnsupportedModalityError for unsupported modality."""
    # More robust error matching (as suggested in review)
    with pytest.raises(UnsupportedModalityError) as exc_info:
        preprocess_dicom_series("dummy_path.dcm", "PET")

    error_message = str(exc_info.value)
    # Check that all supported modalities are mentioned (order doesn't matter)
    assert "CT" in error_message
    assert "MR" in error_message
    assert "MRI" in error_message
    assert "PET" in error_message or "Unsupported modality" in error_message


def test_preprocess_dicom_series_invalid_type():
    """Test preprocess_dicom_series raises ModalityTypeError for non-string modality."""
    with pytest.raises(ModalityTypeError, match=r"modality must be a string, got int"):
        preprocess_dicom_series("dummy_path.dcm", 123)


# ------------------------
# Tests for valid modalities
# ------------------------

@patch("monai.transforms.clinical_preprocessing.get_ct_preprocessing_pipeline")
def test_preprocess_dicom_series_ct(mock_pipeline):
    """Test preprocess_dicom_series successfully runs for CT modality."""
    dummy_output = "ct_processed"
    # Fixed: Use Mock instead of lambda with unused argument (as suggested in review)
    mock_pipeline.return_value = Mock(return_value=dummy_output)
    result = preprocess_dicom_series("dummy_path.dcm", "CT")
    assert result == dummy_output

    # Test lowercase and whitespace variants
    result2 = preprocess_dicom_series("dummy_path.dcm", " ct ")
    assert result2 == dummy_output


@patch("monai.transforms.clinical_preprocessing.get_mri_preprocessing_pipeline")
def test_preprocess_dicom_series_mr(mock_pipeline):
    """Test preprocess_dicom_series successfully runs for MR modality."""
    dummy_output = "mr_processed"
    # Fixed: Use Mock instead of lambda with unused argument (as suggested in review)
    mock_pipeline.return_value = Mock(return_value=dummy_output)
    result = preprocess_dicom_series("dummy_path.dcm", "MR")
    assert result == dummy_output

    # Test lowercase and "MRI" variant
    result2 = preprocess_dicom_series("dummy_path.dcm", "mri")
    assert result2 == dummy_output
