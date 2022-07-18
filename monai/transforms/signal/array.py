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
A collection of transforms for signal operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from typing import List, Optional, Sequence

import numpy as np

from monai.transforms.transform import RandomizableTransform, Transform
from monai.transforms.utils import check_boundaries, paste
from monai.utils import optional_import
from monai.utils.enums import TransformBackends

zoom, has_zoom = optional_import("scipy.ndimage", name="zoom")
resample_poly, has_resample_poly = optional_import("scipy.signal", name="resample_poly")
fft, has_resample_fft = optional_import("scipy.signal", name="resample")
shift, has_shift = optional_import("scipy.ndimage.interpolation", name="shift")
square, has_square = optional_import("scipy.signal", name="square")
iirnotch, has_iirnotch = optional_import("scipy.signal", name="iirnotch")
filtfilt, has_filtfilt = optional_import("scipy.signal", name="filtfilt")
spectrogram, has_spectrogram = optional_import("scipy.signal", name="spectrogram")
central_frequency, has_central_frequency = optional_import("pywt", name="central_frequency")
cwt, has_cwt = optional_import("pywt", name="cwt")


__all__ = [
    "SignalRandDrop",
    "SignalRandScale",
    "SignalRandShift",
    "SignalResample",
    "SignalRandAddSine",
    "SignalRandAddSquarePulse",
    "SignalRandAddGaussianNoise",
    "SignalRandAddSinePartial",
    "SignalRandAddSquarePulsePartial",
    "SignalZeroPad",
    "SignalFillEmpty",
    "SignalRemoveFrequency",
    "SignalShortTimeFourier",
    "SignalContinousWavelet",
]


class SignalResample(Transform):
    """
    Resample signal to the target sampling rate.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        method: str = "interpolation",
        current_sample_rate: Optional[float] = None,
        target_sample_rate: Optional[float] = None,
    ) -> None:
        """
        Args:
            method: which method to be used to resample the signal, default is interpolation.
            Available options : {``"interpolation"``, ``"polynomial"``, ``"fourier"``}
            current_sample_rate: initial sampling rate of the signal
            target_sample_rate: target signal sample rate
        """
        if method is None and method not in ["interpolation", "polynomial", "fourier"]:
            raise ValueError("Incompatible values: method needs to be either interpolation, polynomial or fourier.")
        if current_sample_rate < target_sample_rate:
            raise ValueError(
                "Incompatible target_sampling_rate: target_sampling_rate must be lower current_sampling_rate."
            )
        self.method = method
        self.current_sample_rate = current_sample_rate
        self.target_sample_rate = target_sample_rate

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """

        inputs_channels = signal.shape[0]

        target_length = int(np.round(signal.shape[1] * self.target_sample_rate / self.current_sample_rate))

        if self.method == "interpolation":
            signal = np.stack([zoom(signal[i, :], target_length / signal.shape[1]) for i in range(inputs_channels)])

        elif self.method == "polynomial":
            signal = np.stack(
                [resample_poly(signal[i, :], target_length, signal.shape[1]) for i in range(inputs_channels)]
            )

        elif self.method == "fourier":
            signal = np.stack([fft(signal[i, :], target_length) for i in range(inputs_channels)])

        return signal


class SignalRandShift(RandomizableTransform):
    """
    Apply a random shift on a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self, mode: Optional[str] = "wrap", filling: Optional[float] = 0.0, boundaries: Sequence[float] = (-1.0, 1.0)
    ) -> None:
        """
        Args:
            mode: define how the extension of the input array is done beyond its boundaries, see for more details :
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html.
            filling: value to fill past edges of input if mode is ‘constant’. Default is 0.0. see for mode details :
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html.
            boundaries: list defining lower and upper boundaries for the signal shift, example : ``[-1.0, 1.0]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.filling = filling
        self.mode = mode
        self.boundaries = boundaries

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be shifted
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        length = signal.shape[1]
        shift_idx = round(self.magnitude * length)
        signal = shift(input=signal, mode=self.mode, shift=shift_idx, cval=self.filling)

        return signal


class SignalRandScale(RandomizableTransform):
    """
    Apply a random rescaling on a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, boundaries: Optional[List[float]] = None) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the signal scaling, example : ``[-1.0, 1.0]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be scaled
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        signal = self.magnitude * signal

        return signal


class SignalRandDrop(RandomizableTransform):
    """
    Randomly drop a portion of a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, boundaries: Optional[List[float]] = None) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the signal drop,
            lower and upper values need to be positive example : ``[0.2, 0.6]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be dropped
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])

        length = signal.shape[len(signal.shape) - 1]

        mask = np.zeros(round(self.magnitude * length))
        loc = np.random.choice(range(length))
        signal = paste(signal, mask, (loc,))

        return signal


class SignalRandAddSine(RandomizableTransform):
    """
    Add a random sinusoidal signal to the input signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, boundaries: Optional[List[float]] = None, frequencies: Optional[List[float]] = None) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the sinusoidal magnitude,
            lower and upper values need to be positive example : ``[0.2, 0.6]``
            frequencies: list defining lower and upper frequencies for sinusoidal
            signal generation example : ``[0.001, 0.02]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries
        self.frequencies = frequencies

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to which sinusoidal signal will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

        length = signal.shape[1]

        time = np.arange(0, length, 1)
        sine = self.magnitude * np.sin(self.freqs * time)

        return signal + sine


class SignalRandAddSquarePulse(RandomizableTransform):
    """
    Add a random square pulse signal to the input signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, boundaries: Optional[List[float]] = None, frequencies: Optional[List[float]] = None) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the square pulse magnitude,
            lower and upper values need to be positive example : ``[0.2, 0.6]``
            frequencies: list defining lower and upper frequencies for the square pulse
            signal generation example : ``[0.001, 0.02]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries
        self.frequencies = frequencies

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to which square pulse will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

        length = signal.shape[1]

        time = np.arange(0, length, 1)
        squaredpulse = self.magnitude * square(self.freqs * time)

        return signal + squaredpulse


class SignalRandAddSinePartial(RandomizableTransform):
    """
    Add a random partial sinusoidal signal to the input signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        boundaries: Optional[List[float]] = None,
        frequencies: Optional[List[float]] = None,
        fraction: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the sinusoidal magnitude,
            lower and upper values need to be positive example : ``[0.2, 0.6]``
            frequencies: list defining lower and upper frequencies for sinusoidal
            signal generation example : ``[0.001, 0.02]``
            fraction: list defining lower and upper boundaries for partial signal generation
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries
        self.frequencies = frequencies
        self.fraction = fraction

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to which a partial sinusoidal signal
            will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.fracs = self.R.uniform(low=self.fraction[0], high=self.fraction[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

        length = signal.shape[len(signal.shape) - 1]

        time_partial = np.arange(0, round(self.fracs * length), 1)
        sine_partial = self.magnitude * np.sin(self.freqs * time_partial)

        loc = np.random.choice(range(length))
        signal = paste(signal, sine_partial, (loc,))

        return signal


class SignalRandAddGaussianNoise(RandomizableTransform):
    """
    Add a random gaussian noise to the input signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, boundaries: Optional[List[float]] = None) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the signal magnitude,
            lower and upper values need to be positive example : ``[0.2, 0.6]``
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to which gaussian noise will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        length = signal.shape[1]
        gaussiannoise = self.magnitude * np.random.normal(size=length)

        signal = signal + gaussiannoise

        return signal


class SignalRandAddSquarePulsePartial(RandomizableTransform):
    """
    Add a random partial square pulse to a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        boundaries: Optional[List[float]] = None,
        frequencies: Optional[List[float]] = None,
        fraction: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            boundaries: list defining lower and upper boundaries for the square pulse magnitude,
            lower and upper values need to be positive example : ``[0.2, 0.6]``
            frequencies: list defining lower and upper frequencies for square pulse
            signal generation example : ``[0.001, 0.02]``
            fraction: list defining lower and upper boundaries for partial square pulse generation
        """
        super().__init__()
        check_boundaries(boundaries)
        self.boundaries = boundaries
        self.frequencies = frequencies
        self.fraction = fraction

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to which a partial square pulse will be added
        """
        self.randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.fracs = self.R.uniform(low=self.fraction[0], high=self.fraction[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

        length = signal.shape[len(signal.shape) - 1]

        time_partial = np.arange(0, round(self.fracs * length), 1)
        squaredpulse_partial = self.magnitude * square(self.freqs * time_partial)

        loc = np.random.choice(range(length))
        signal = paste(signal, squaredpulse_partial, (loc,))

        return signal


class SignalFillEmpty(Transform):
    """
    replace empty part of a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, replacement: Optional[float] = 0.0) -> None:
        """
        Args:
            replacement: value to replace nan items in signal
        """
        super().__init__()
        self.replacement = replacement

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: signal to be filled
        """

        return np.nan_to_num(signal, nan=self.replacement)


class SignalRemoveFrequency(Transform):
    """
    Remove a frequency from a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        frequency: Optional[float] = None,
        quality_factor: Optional[float] = None,
        sampling_freq: Optional[float] = None,
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

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: signal to be frequency removed
        """

        b_notch, a_notch = iirnotch(self.frequency, self.quality_factor, self.sampling_freq)
        y_notched = filtfilt(b_notch, a_notch, signal)

        return y_notched


class SignalShortTimeFourier(Transform):
    """
    Generate short time Fourier transform of a signal.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self, frequency: Optional[int] = None, nperseg: Optional[int] = None, noverlap: Optional[int] = None
    ) -> None:
        """
        Args:
            frequency: signal frequency
            nperseg: length of each segment for Short Time Fourier analysis
            noverlap: overlaping section between each segments
        """
        super().__init__()
        self.frequency = frequency
        self.nperseg = nperseg
        self.noverlap = noverlap

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: signal to be processed with Short Time Fourier transform
        """
        f, t, sxx = spectrogram(signal, fs=self.frequency, nperseg=self.nperseg, noverlap=self.noverlap)

        sxx = np.transpose(sxx, [0, 2, 1])

        sxx = np.abs(sxx)
        mask = sxx > 0
        sxx[mask] = np.log(sxx[mask])

        sxx = (sxx - np.mean(sxx)) / np.std(sxx)

        sx_norm = np.transpose(sxx)

        sx_norm = np.transpose(sx_norm, [2, 0, 1])

        return sx_norm


class SignalContinousWavelet(Transform):
    """
    Generate continuous wavelet transform of a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, type: str = "mexh", length: Optional[int] = None, frequency: Optional[int] = None) -> None:
        """
        Args:
            type: mother wavelet type.
            Available options are: {``"mexh"``, ``"morl"``, ``"cmorB-C"``, , ``"gausP"``}
            see : https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
            length: expected length
            frequency: signal frequency
        """
        super().__init__()
        self.frequency = frequency
        self.length = length
        self.type = type

    def __call__(self, signal: np.ndarray) -> np.ndarray:
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
