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

import tempfile
from pathlib import Path

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


def test_unsupported_modality():
    """Test unsupported modality."""
    with pytest.raises(UnsupportedModalityError) as exc:
        preprocess_dicom_series("dummy", "PET")

    msg = str(exc.value)
    assert "Unsupported modality" in msg
    assert "CT" in msg
    assert "MR" in msg
    assert "MRI" in msg


def test_modality_case_insensitivity():
    """Test case-insensitive modality handling."""
    # Test each case variation
    for modality in ["CT", "ct", "Ct", "CT ", "MR", "mr", "MRI", "mri", " MrI "]:
        # Just test that the function doesn't raise modality-related errors
        # We're not testing actual image loading, just modality parsing
        try:
            # This will fail on file loading, but not on modality parsing
            preprocess_dicom_series("non_existent_file.dcm", modality)
        except (ModalityTypeError, UnsupportedModalityError):
            pytest.fail(f"Modality {modality!r} should be accepted")
        except FileNotFoundError:
            # This is expected - the file doesn't exist, but modality parsing worked
            pass
        except Exception:
            # Any other error is fine for this test
            pass


def test_preprocess_dicom_series_integration(tmp_path):
    """Integration test with dummy NIfTI file."""
    # Create a dummy NIfTI file for testing
    dummy_data = np.random.randn(64, 64, 64).astype(np.float32)
    test_file = tmp_path / "test.nii.gz"

    write_nifti(dummy_data, test_file)

    # Test with each modality
    for modality in ["CT", "MRI"]:
        result = preprocess_dicom_series(str(test_file), modality)
        assert result is not None
        assert hasattr(result, "shape")