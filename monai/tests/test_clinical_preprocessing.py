import numpy as np
import pytest

from monai.transforms import ScaleIntensityRange, NormalizeIntensity


def test_ct_windowing_range_and_shape():
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
