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

from unittest.mock import Mock, patch

import numpy as np
import pytest

from monai.data import write_nifti
from monai.transforms import EnsureChannelFirst, LoadImage, NormalizeIntensity, ScaleIntensityRange
from monai.transforms.clinical_preprocessing import (
    ModalityTypeError,
    UnsupportedModalityError,
    get_ct_preprocessing_pipeline,
    get_mri_preprocessing_pipeline,
    preprocess_dicom_series,
    preprocess_medical_image,
)


def test_ct_preprocessing_pipeline_structure():
    """Test CT pipeline structure."""
    pipeline = get_ct_preprocessing_pipeline()
    transforms = pipeline.transforms

    assert len(transforms) == 3
    assert isinstance(transforms[0], LoadImage)
    assert transforms[0].image_only is True
    assert transforms[1].__class__ is EnsureChannelFirst
    assert isinstance(transforms[2], ScaleIntensityRange)

    scale = transforms[2]
    assert scale.a_min == -1000
    assert scale.a_max == 400
    assert scale.b_min == 0.0
    assert scale.b_max == 1.0
    assert scale.clip is True


def test_mri_preprocessing_pipeline_structure():
    """Test MRI pipeline structure."""
    pipeline = get_mri_preprocessing_pipeline()
    transforms = pipeline.transforms

    assert len(transforms) == 3
    assert isinstance(transforms[0], LoadImage)
    assert transforms[0].image_only is True
    assert transforms[1].__class__ is EnsureChannelFirst
    assert isinstance(transforms[2], NormalizeIntensity)
    assert transforms[2].nonzero is True


def test_invalid_modality_type():
    """Test non-string modality input."""
    with pytest.raises(ModalityTypeError) as exc:
        preprocess_dicom_series("dummy", 123)

    assert "modality must be a string" in str(exc.value)

    with pytest.raises(ModalityTypeError) as exc:
        preprocess_medical_image("dummy", None)

    assert "modality must be a string" in str(exc.value)


def test_unsupported_modality():
    """Test unsupported modality."""
    with pytest.raises(UnsupportedModalityError) as exc:
        preprocess_dicom_series("dummy", "PET")

    msg = str(exc.value)
    assert "Unsupported modality" in msg
    assert "PET" in msg
    assert "CT" in msg
    assert "MR" in msg
    assert "MRI" in msg

    with pytest.raises(UnsupportedModalityError) as exc:
        preprocess_medical_image("dummy", "PET")

    msg = str(exc.value)
    assert "Unsupported modality" in msg
    assert "PET" in msg
    assert "CT" in msg
    assert "MR" in msg
    assert "MRI" in msg


@patch("monai.transforms.clinical_preprocessing.LoadImage")
def test_modality_case_insensitivity(mock_load):
    """Test case-insensitive modality handling with whitespace trimming."""
    mock_load.return_value = Mock(return_value=Mock())

    test_cases = [
        ("CT", True),
        ("ct", True),
        ("Ct", True),
        ("CT ", True),
        (" CT", True),
        ("MR", True),
        ("mr", True),
        ("MRI", True),
        ("mri", True),
        (" MrI ", True),
    ]

    for modality, _ in test_cases:
        result = preprocess_dicom_series("dummy.dcm", modality)
        assert result is not None, f"Failed for modality: '{modality}'"
        result2 = preprocess_medical_image("dummy.dcm", modality)
        assert result2 is not None, f"preprocess_medical_image failed for modality: '{modality}'"


@patch("monai.transforms.clinical_preprocessing.LoadImage")
def test_mr_modality_distinct(mock_load):
    """Test MR modality is handled separately from MRI."""
    mock_load.return_value = Mock(return_value=Mock())
    result = preprocess_dicom_series("dummy.dcm", "MR")
    assert result is not None
    result2 = preprocess_medical_image("dummy.dcm", "MR")
    assert result2 is not None


@patch("monai.transforms.clinical_preprocessing.LoadImage")
def test_edge_cases(mock_load):
    """Test edge cases for modality input."""
    mock_load.return_value = Mock(return_value=Mock())

    with pytest.raises(UnsupportedModalityError):
        preprocess_dicom_series("dummy.dcm", "")

    with pytest.raises(UnsupportedModalityError):
        preprocess_dicom_series("dummy.dcm", "   ")

    long_modality = "CT" * 100
    with pytest.raises(UnsupportedModalityError):
        preprocess_dicom_series("dummy.dcm", long_modality)


def test_preprocess_dicom_series_integration(tmp_path):
    """Integration test with dummy NIfTI file."""
    dummy_data = np.random.randn(64, 64, 64).astype(np.float32)
    test_file = tmp_path / "test.nii.gz"

    write_nifti(dummy_data, test_file)

    for modality in ["CT", "MR", "MRI"]:
        result = preprocess_dicom_series(str(test_file), modality)
        assert result is not None
        assert hasattr(result, "shape")
        result2 = preprocess_medical_image(str(test_file), modality)
        assert result2 is not None
        assert hasattr(result2, "shape")