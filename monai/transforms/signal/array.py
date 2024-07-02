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
"""
A collection of transforms for signal operations.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import RandomizableTransform, Transform
from monai.transforms.utils import check_boundaries, paste, squarepulse
from monai.utils import optional_import
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_tensor

shift, has_shift = optional_import("scipy.ndimage", name="shift")
iirnotch, has_iirnotch = optional_import("scipy.signal", name="iirnotch")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)  # project-monai/monai#5204
    filtfilt, has_filtfilt = optional_import("torchaudio.functional", name="filtfilt")
central_frequency, has_central_frequency = optional_import("pywt", name="central_frequency")
cwt, has_cwt = optional_import("pywt", name="cwt")

__all__ = [
    "SignalRandDrop",
    "SignalRandScale",
    "SignalRandShift",
    "SignalRandAddSine",
    "SignalRandAddSquarePulse",
    "SignalRandAddGaussianNoise",
    "SignalRandAddSinePartial",
    "SignalRandAddSquarePulsePartial",
    "SignalFillEmpty",
    "SignalRemoveFrequency",
    "SignalContinuousWavelet",
]


class SignalRandShift(RandomizableTransform):
    """
    Apply a random shift on a signal
    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self, mode: str | None = "wrap", filling: float | None = 0.0, boundaries: Sequence[float] = (-1.0, 1.0)
    ) -> None:
        """
        Args:
            mode: define how the extension of the input array is done beyond its boundaries, see for more details :
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html.
            filling: value to fill past edges of input if mode is ‘constant’. Default is 0.0. see for mode details :
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html.
            boundaries: list defining lower and upper boundaries for the signal shift, default : ``[-1.0, 1.0]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.filling = filling
        self.mode = mode
        self.boundaries = boundaries

    def __call__(self, signal: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            signal: input 1 dimension signal to be shifted
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        length = signal.shape[1]
        shift_idx = round(self.magnitude * length)
        sig = convert_data_type(signal, np.ndarray)[0]
        signal = convert_to_tensor(shift(input=sig, mode=self.mode, shift=shift_idx, cval=self.filling))
        return signal


class SignalRandScale(RandomizableTransform):
    """
    Apply a random rescaling on a signal
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, boundaries: Sequence[float] = (-1.0, 1.0)) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the signal scaling, default : ``[-1.0, 1.0]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries

    def __call__(self, signal: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            signal: input 1 dimension signal to be scaled
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        signal = convert_to_tensor(self.magnitude * signal)

        return signal


class SignalRandDrop(RandomizableTransform):
    """
    Randomly drop a portion of a signal
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, boundaries: Sequence[float] = (0.0, 1.0)) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the signal drop,
            lower and upper values need to be positive default : ``[0.0, 1.0]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries

    def __call__(self, signal: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            signal: input 1 dimension signal to be dropped
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])

        length = signal.shape[-1]
        mask = torch.zeros(round(self.magnitude * length))
        trange = torch.arange(length)
        loc = trange[torch.randint(0, trange.size(0), (1,))]
        signal = convert_to_tensor(paste(signal, mask, (loc,)))

        return signal


class SignalRandAddSine(RandomizableTransform):
    """
    Add a random sinusoidal signal to the input signal
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, boundaries: Sequence[float] = (0.1, 0.3), frequencies: Sequence[float] = (0.001, 0.02)) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the sinusoidal magnitude,
                lower and upper values need to be positive ,default : ``[0.1, 0.3]``
            frequencies: list defining lower and upper frequencies for sinusoidal
                signal generation ,default : ``[0.001, 0.02]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries
        self.frequencies = frequencies

    def __call__(self, signal: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            signal: input 1 dimension signal to which sinusoidal signal will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

        length = signal.shape[1]

        time = np.arange(0, length, 1)
        data = convert_to_tensor(self.freqs * time)
        sine = self.magnitude * torch.sin(data)
        signal = convert_to_tensor(signal) + sine

        return signal


class SignalRandAddSquarePulse(RandomizableTransform):
    """
    Add a random square pulse signal to the input signal
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, boundaries: Sequence[float] = (0.01, 0.2), frequencies: Sequence[float] = (0.001, 0.02)) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the square pulse magnitude,
                lower and upper values need to be positive , default : ``[0.01, 0.2]``
            frequencies: list defining lower and upper frequencies for the square pulse
                signal generation , default : ``[0.001, 0.02]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries
        self.frequencies = frequencies

    def __call__(self, signal: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            signal: input 1 dimension signal to which square pulse will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

        length = signal.shape[1]

        time = np.arange(0, length, 1)
        squaredpulse = self.magnitude * squarepulse(self.freqs * time)
        signal = convert_to_tensor(signal) + squaredpulse

        return signal


class SignalRandAddSinePartial(RandomizableTransform):
    """
    Add a random partial sinusoidal signal to the input signal
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        boundaries: Sequence[float] = (0.1, 0.3),
        frequencies: Sequence[float] = (0.001, 0.02),
        fraction: Sequence[float] = (0.01, 0.2),
    ) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the sinusoidal magnitude,
                lower and upper values need to be positive , default : ``[0.1, 0.3]``
            frequencies: list defining lower and upper frequencies for sinusoidal
                signal generation , default : ``[0.001, 0.02]``
            fraction: list defining lower and upper boundaries for partial signal generation
                default : ``[0.01, 0.2]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries
        self.frequencies = frequencies
        self.fraction = fraction

    def __call__(self, signal: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            signal: input 1 dimension signal to which a partial sinusoidal signal
            will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.fracs = self.R.uniform(low=self.fraction[0], high=self.fraction[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

        length = signal.shape[-1]

        time_partial = np.arange(0, round(self.fracs * length), 1)
        data = convert_to_tensor(self.freqs * time_partial)
        sine_partial = self.magnitude * torch.sin(data)

        loc = np.random.choice(range(length))
        signal = paste(signal, sine_partial, (loc,))

        return signal


class SignalRandAddGaussianNoise(RandomizableTransform):
    """
    Add a random gaussian noise to the input signal
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, boundaries: Sequence[float] = (0.001, 0.02)) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the signal magnitude,
                default : ``[0.001,0.02]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries

    def __call__(self, signal: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            signal: input 1 dimension signal to which gaussian noise will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        length = signal.shape[1]
        gaussiannoise = self.magnitude * torch.randn(length)

        signal = convert_to_tensor(signal) + gaussiannoise

        return signal


class SignalRandAddSquarePulsePartial(RandomizableTransform):
    """
    Add a random partial square pulse to a signal
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        boundaries: Sequence[float] = (0.01, 0.2),
        frequencies: Sequence[float] = (0.001, 0.02),
        fraction: Sequence[float] = (0.01, 0.2),
    ) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the square pulse magnitude,
                lower and upper values need to be positive , default : ``[0.01, 0.2]``
            frequencies: list defining lower and upper frequencies for square pulse
                signal generation example : ``[0.001, 0.02]``
            fraction: list defining lower and upper boundaries for partial square pulse generation
                default: ``[0.01, 0.2]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries
        self.frequencies = frequencies
        self.fraction = fraction

    def __call__(self, signal: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            signal: input 1 dimension signal to which a partial square pulse will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.fracs = self.R.uniform(low=self.fraction[0], high=self.fraction[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

        length = signal.shape[-1]

        time_partial = np.arange(0, round(self.fracs * length), 1)
        squaredpulse_partial = self.magnitude * squarepulse(self.freqs * time_partial)

        loc = np.random.choice(range(length))
        signal = paste(signal, squaredpulse_partial, (loc,))

        return signal


class SignalFillEmpty(Transform):
    """
    replace empty part of a signal (NaN)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, replacement: float = 0.0) -> None:
        """
        Args:
            replacement: value to replace nan items in signal
        """
        super().__init__()
        self.replacement = replacement

    def __call__(self, signal: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            signal: signal to be filled
        """
        signal = torch.nan_to_num(convert_to_tensor(signal, track_meta=True), nan=self.replacement)
        return signal


class SignalRemoveFrequency(Transform):
    """
    Remove a frequency from a signal
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self, frequency: float | None = None, quality_factor: float | None = None, sampling_freq: float | None = None
    ) -> None:
        """
        Args:
            frequency: frequency to be removed from the signal
            quality_factor: quality factor for notch filter
                see : https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
            sampling_freq: sampling frequency of the input signal
        """
        super().__init__()
        self.frequency = frequency
        self.quality_factor = quality_factor
        self.sampling_freq = sampling_freq

    def __call__(self, signal: np.ndarray) -> Any:
        """
        Args:
            signal: signal to be frequency removed
        """
        b_notch, a_notch = convert_to_tensor(
            iirnotch(self.frequency, self.quality_factor, self.sampling_freq), dtype=torch.float
        )
        y_notched = filtfilt(convert_to_tensor(signal), a_notch, b_notch)

        return y_notched


class SignalContinuousWavelet(Transform):
    """
    Generate continuous wavelet transform of a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, type: str = "mexh", length: float = 125.0, frequency: float = 500.0) -> None:
        """
        Args:
            type: mother wavelet type.
                Available options are: {``"mexh"``, ``"morl"``, ``"cmorB-C"``, , ``"gausP"``}
            see : https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
            length: expected length, default ``125.0``
            frequency: signal frequency, default ``500.0``
        """
        super().__init__()
        self.frequency = frequency
        self.length = length
        self.type = type

    def __call__(self, signal: np.ndarray) -> Any:
        """
        Args:
            signal: signal for which to generate continuous wavelet transform
        """
        mother_wavelet = self.type
        spread = np.arange(1, self.length + 1, 1)
        scales = central_frequency(mother_wavelet) * self.frequency / spread

        coeffs, _ = cwt(signal, scales, mother_wavelet, 1.0 / self.frequency)

        coeffs = np.transpose(coeffs, [1, 0, 2])

        return coeffs
