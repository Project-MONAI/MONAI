"""Unit tests for clinical DICOM preprocessing utilities."""

import numpy as np
from unittest.mock import patch, MagicMock
import pytest

from monai.transforms import ScaleIntensityRange, NormalizeIntensity
from monai.transforms.clinical_preprocessing import (
    get_ct_preprocessing_pipeline,
    get_mri_preprocessing_pipeline,
    preprocess_dicom_series,
)


def test_ct_windowing_range_and_shape_direct():
    """Test ScaleIntensityRange transform on sample CT data."""
    rng = np.random.default_rng(0)
    sample_ct = rng.integers(-1024, 2048, size=(64, 64, 64), dtype=np.int16)
    transform = ScaleIntensityRange(a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True)
    output = np.asarray(transform(sample_ct))

    assert output.shape == sample_ct.shape
    assert np.isfinite(output).all()
    assert output.min() >= -1e-6
    assert output.max() <= 1.0 + 1e-6


def test_mri_normalization_mean_std_direct():
    """Test NormalizeIntensity transform on sample MRI data."""
    rng = np.random.default_rng(0)
    sample_mri = rng.random((64, 64, 64), dtype=np.float32)
    transform = NormalizeIntensity(nonzero=True)
    output = np.asarray(transform(sample_mri))

    assert output.shape == sample_mri.shape
    assert np.isclose(float(output.mean()), 0.0, atol=0.1)
    assert np.isclose(float(output.std()), 1.0, atol=0.1)


@patch("monai.transforms.clinical_preprocessing.LoadImage")
def test_ct_pipeline(mock_loadimage):
    """Test get_ct_preprocessing_pipeline returns correct transform sequence."""
    pipeline = get_ct_preprocessing_pipeline()
    assert len(pipeline.transforms) == 3
    assert pipeline.transforms[0].__class__.__name__ == "LoadImage"
    assert pipeline.transforms[1].__class__.__name__ == "EnsureChannelFirst"
    assert pipeline.transforms[2].__class__.__name__ == "ScaleIntensityRange"


@patch("monai.transforms.clinical_preprocessing.LoadImage")
def test_mri_pipeline(mock_loadimage):
    """Test get_mri_preprocessing_pipeline returns correct transform sequence."""
    pipeline = get_mri_preprocessing_pipeline()
    assert len(pipeline.transforms) == 3
    assert pipeline.transforms[0].__class__.__name__ == "LoadImage"
    assert pipeline.transforms[1].__class__.__name__ == "EnsureChannelFirst"
    assert pipeline.transforms[2].__class__.__name__ == "NormalizeIntensity"


@patch("monai.transforms.clinical_preprocessing.get_ct_preprocessing_pipeline")
def test_preprocess_dicom_series_ct(mock_pipeline):
    """Test preprocess_dicom_series with CT modality."""
    mock_transform = MagicMock()
    mock_pipeline.return_value = mock_transform
    preprocess_dicom_series("dummy_path.dcm", "CT")
    mock_pipeline.assert_called_once()
    mock_transform.assert_called_once_with("dummy_path.dcm")


@patch("monai.transforms.clinical_preprocessing.get_ct_preprocessing_pipeline")
def test_preprocess_dicom_series_ct_lowercase(mock_pipeline):
    """Test preprocess_dicom_series with lowercase CT modality."""
    mock_transform = MagicMock()
    mock_pipeline.return_value = mock_transform
    preprocess_dicom_series("dummy_path.dcm", "ct")
    mock_pipeline.assert_called_once()
    mock_transform.assert_called_once_with("dummy_path.dcm")


@patch("monai.transforms.clinical_preprocessing.get_mri_preprocessing_pipeline")
def test_preprocess_dicom_series_mri(mock_pipeline):
    """Test preprocess_dicom_series with MRI modality."""
    mock_transform = MagicMock()
    mock_pipeline.return_value = mock_transform
    preprocess_dicom_series("dummy_path.dcm", "MRI")
    mock_pipeline.assert_called_once()
    mock_transform.assert_called_once_with("dummy_path.dcm")


@patch("monai.transforms.clinical_preprocessing.get_mri_preprocessing_pipeline")
def test_preprocess_dicom_series_mr(mock_pipeline):
    """Test preprocess_dicom_series with MR modality."""
    mock_transform = MagicMock()
    mock_pipeline.return_value = mock_transform
    preprocess_dicom_series("dummy_path.dcm", "MR")
    mock_pipeline.assert_called_once()
    mock_transform.assert_called_once_with("dummy_path.dcm")


def test_preprocess_dicom_series_invalid_modality():
    """Test preprocess_dicom_series raises ValueError for unsupported modality."""
    with pytest.raises(ValueError) as exc:
        preprocess_dicom_series("dummy_path.dcm", "PET")
    assert "Unsupported modality" in str(exc.value)
    assert "PET" in str(exc.value)


def test_preprocess_dicom_series_invalid_type():
    """Test preprocess_dicom_series raises TypeError for non-string modality."""
    with pytest.raises(TypeError) as exc:
        preprocess_dicom_series("dummy_path.dcm", 123)
    assert "modality must be a string" in str(exc.value)


def test_preprocess_dicom_series_none_modality():
    """Test preprocess_dicom_series raises TypeError for None modality."""
    with pytest.raises(TypeError) as exc:
        preprocess_dicom_series("dummy_path.dcm", None)
    assert "modality must be a string" in str(exc.value)


@patch("monai.transforms.clinical_preprocessing.get_ct_preprocessing_pipeline")
def test_preprocess_dicom_series_whitespace(mock_pipeline):
    """Test preprocess_dicom_series handles whitespace in modality."""
    mock_transform = MagicMock()
    mock_pipeline.return_value = mock_transform
    preprocess_dicom_series("dummy_path.dcm", "  CT  ")
    mock_pipeline.assert_called_once()
    mock_transform.assert_called_once_with("dummy_path.dcm")
