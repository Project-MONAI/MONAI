import numpy as np

from monai.transforms import ScaleIntensityRange, NormalizeIntensity
from monai.transforms.clinical_preprocessing import (
    get_ct_preprocessing_pipeline,
    get_mri_preprocessing_pipeline,
    preprocess_dicom_series,
)
from unittest.mock import patch, MagicMock


def test_ct_windowing_range_and_shape():
    """Test CT windowing transform parameters."""
    rng = np.random.default_rng(0)

    sample_ct = rng.integers(
        -1024, 2048, size=(64, 64, 64), dtype=np.int16
    )

    transform = ScaleIntensityRange(
        a_min=-1000,
        a_max=400,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )

    output = transform(sample_ct)
    output = np.asarray(output)

    assert output.shape == sample_ct.shape
    assert np.isfinite(output).all()
    assert output.min() >= -1e-6
    assert output.max() <= 1.0 + 1e-6


def test_mri_normalization_mean_std():
    """Test MRI normalization transform."""
    rng = np.random.default_rng(0)

    sample_mri = rng.random((64, 64, 64), dtype=np.float32)

    transform = NormalizeIntensity(nonzero=True)

    output = transform(sample_mri)
    output = np.asarray(output)

    mean_val = float(output.mean())
    std_val = float(output.std())

    assert output.shape == sample_mri.shape
    assert np.isclose(mean_val, 0.0, atol=0.1)
    assert np.isclose(std_val, 1.0, atol=0.1)


def test_ct_preprocessing_pipeline():
    """Test CT preprocessing pipeline returns expected transform composition."""
    pipeline = get_ct_preprocessing_pipeline()

    assert hasattr(pipeline, 'transforms')
    assert len(pipeline.transforms) == 3
    assert pipeline.transforms[0].__class__.__name__ == 'LoadImage'
    assert pipeline.transforms[1].__class__.__name__ == 'EnsureChannelFirst'
    assert pipeline.transforms[2].__class__.__name__ == 'ScaleIntensityRange'


def test_mri_preprocessing_pipeline():
    """Test MRI preprocessing pipeline returns expected transform composition."""
    pipeline = get_mri_preprocessing_pipeline()

    assert hasattr(pipeline, 'transforms')
    assert len(pipeline.transforms) == 3
    assert pipeline.transforms[0].__class__.__name__ == 'LoadImage'
    assert pipeline.transforms[1].__class__.__name__ == 'EnsureChannelFirst'
    assert pipeline.transforms[2].__class__.__name__ == 'NormalizeIntensity'


@patch('monai.transforms.clinical_preprocessing.get_ct_preprocessing_pipeline')
def test_preprocess_dicom_series_ct(mock_pipeline):
    """Test preprocess_dicom_series with CT modality."""
    mock_transform = MagicMock()
    mock_pipeline.return_value = mock_transform

    preprocess_dicom_series("dummy_path.dcm", "CT")

    mock_pipeline.assert_called_once()
    mock_transform.assert_called_once_with("dummy_path.dcm")


@patch('monai.transforms.clinical_preprocessing.get_ct_preprocessing_pipeline')
def test_preprocess_dicom_series_ct_lowercase(mock_pipeline):
    """Test preprocess_dicom_series with CT modality in lowercase."""
    mock_transform = MagicMock()
    mock_pipeline.return_value = mock_transform

    preprocess_dicom_series("dummy_path.dcm", "ct")

    mock_pipeline.assert_called_once()
    mock_transform.assert_called_once_with("dummy_path.dcm")


@patch('monai.transforms.clinical_preprocessing.get_mri_preprocessing_pipeline')
def test_preprocess_dicom_series_mri(mock_pipeline):
    """Test preprocess_dicom_series with MRI modality."""
    mock_transform = MagicMock()
    mock_pipeline.return_value = mock_transform

    preprocess_dicom_series("dummy_path.dcm", "MRI")

    mock_pipeline.assert_called_once()
    mock_transform.assert_called_once_with("dummy_path.dcm")


@patch('monai.transforms.clinical_preprocessing.get_mri_preprocessing_pipeline')
def test_preprocess_dicom_series_mr(mock_pipeline):
    """Test preprocess_dicom_series with MR modality."""
    mock_transform = MagicMock()
    mock_pipeline.return_value = mock_transform

    preprocess_dicom_series("dummy_path.dcm", "MR")

    mock_pipeline.assert_called_once()
    mock_transform.assert_called_once_with("dummy_path.dcm")


def test_preprocess_dicom_series_invalid_modality():
    """Test preprocess_dicom_series raises ValueError for unsupported modality."""
    try:
        preprocess_dicom_series("dummy_path.dcm", "PET")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_message = str(e)
        assert "Unsupported modality" in error_message
        assert "PET" in error_message
        assert "CT, MR, MRI" in error_message


def test_preprocess_dicom_series_invalid_type():
    """Test preprocess_dicom_series raises TypeError for non-string modality."""
    try:
        preprocess_dicom_series("dummy_path.dcm", 123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        error_message = str(e)
