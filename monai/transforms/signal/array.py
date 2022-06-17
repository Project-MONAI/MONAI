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

from typing import Optional, Tuple

import numpy as np

from monai.transforms.transform import RandomizableTransform, Transform
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
    "SignalNormalize",
    "SignalStandardize",
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
        self, method: str = "interpolation", current_sample_rate: int = 500, target_sample_rate: int = 250
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
        if len(signal.shape) > 1:
            inputs_channels = signal.shape[0]
        else:
            inputs_channels = 1
            signal = np.expand_dims(signal, axis=0)

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
        self,
        v: Optional[float] = 1.0,
        mode: Optional[str] = "wrap",
        filling: Optional[float] = 0.0,
        boundaries: Tuple[float, float] = Tuple[None, None],
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            v: scaling factor
            mode: define how the extension of the input array is done beyond its boundaries, see for more details :
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html.
            filling: value to fill past edges of input if mode is ‘constant’. Default is 0.0. see for mode details :
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html.
            boundaries: list defining lower and upper boundaries for the signal shift, example : ``[-1.0, 1.0]``
        """
        super().__init__()
        if boundaries is None or None in boundaries:
            raise ValueError("Incompatible values: boundaries needs to be a list of float.")
        self.filling = filling
        self.mode = mode
        self.v = v
        self.boundaries = boundaries

    def _randomize(self):
        super().randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        self._randomize()
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)
        length = signal.shape[1]
        factor = self.v * self.magnitude
        shift_idx = round(factor * length)
        signal = shift(input=signal, mode=self.mode, shift=shift_idx, cval=self.filling)

        # if len(signal.shape) > 1:
        #     signal = np.squeeze(signal)

        return signal


class SignalRandScale(RandomizableTransform):
    """
    Apply a random rescaling on a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self, v: Optional[float] = 1.0, boundaries: Tuple[float, float] = Tuple[None, None], *args, **kwargs
    ) -> None:
        """
        Args:
            v: scaling factor
            boundaries: list defining lower and upper boundaries for the signal shift, example : ``[-1.0, 1.0]``
        """
        super().__init__()
        if boundaries is None or None in boundaries:
            raise ValueError("Incompatible values: boundaries needs to be a list of float.")
        self.v = v
        self.boundaries = boundaries

    def _randomize(self):
        super().randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        self._randomize()
        factor = self.v * self.magnitude
        signal = factor * signal
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)
        return signal


class SignalRandDrop(RandomizableTransform):
    """
    Randomly drop a portion of a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self, v: Optional[float] = 1.0, boundaries: Tuple[float, float] = Tuple[None, None], *args, **kwargs
    ) -> None:
        """
        Args:
            v: scaling factor
            boundaries: list defining lower and upper boundaries for the signal drop, lower and upper values need to be positive example : ``[0.2, 0.6]``
        """
        super().__init__()
        if boundaries is None or None in boundaries:
            raise ValueError("Incompatible values: boundaries needs to be a list of float.")
        if (boundaries is None) or (None in boundaries) or (any(x < 0 for x in boundaries)):
            raise ValueError("Incompatible values: boundaries needs to be a list of positive float.")

        self.v = v
        self.boundaries = boundaries

    def _randomize(self):
        super().randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])

    def _paste_slices(self, tup):
        pos, w, max_w = tup
        max_w = max_w.shape[len(max_w.shape) - 1]
        wall_min = max(pos, 0)
        wall_max = min(pos + w, max_w)
        block_min = -min(pos, 0)
        block_max = max_w - max(pos + w, max_w)
        block_max = block_max if block_max != 0 else None
        return slice(wall_min, wall_max), slice(block_min, block_max)

    def _paste(self, wall, block, loc):
        loc_zip = zip(loc, block.shape, wall)
        wall_slices, block_slices = zip(*map(self._paste_slices, loc_zip))

        wall[:, wall_slices[0]] = block[block_slices[0]]

        if wall.shape[0] == 1:
            wall = wall.squeeze()
        return wall

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        self._randomize()
        if len(signal.shape) > 1:
            length = signal.shape[len(signal.shape) - 1]
        else:
            signal = signal[np.newaxis, :]
            length = signal.shape[1]

        factor = self.v * self.magnitude
        mask = np.zeros(round(factor * length))
        loc = np.random.choice(range(length))
        signal = self._paste(signal, mask, (loc,))

        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)

        return signal


class SignalRandAddSine(RandomizableTransform):
    """
    Add a random sinusoidal signal to the input signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        v: Optional[float] = 1.0,
        boundaries: Tuple[float, float] = Tuple[None, None],
        frequencies: Tuple[float, float] = Tuple[None, None],
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            v: scaling factor
            boundaries: list defining lower and upper boundaries for the signal drop, lower and upper values need to be positive example : ``[0.2, 0.6]``
            frequencies: list defining lower and upper frequencies for sinusoidal signal generation example : ``[0.001, 0.02]``
        """
        super().__init__()
        if boundaries is None or None in boundaries:
            raise ValueError("Incompatible values: boundaries needs to be a list of float.")
        if (boundaries is None) or (None in boundaries) or (any(x < 0 for x in boundaries)):
            raise ValueError("Incompatible values: boundaries needs to be a list of positive float.")

        self.v = v
        self.boundaries = boundaries
        self.frequencies = frequencies

    def _randomize(self):
        super().randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        self._randomize()
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)
        length = signal.shape[1]

        time = np.arange(0, length, 1)
        sine = self.magnitude * np.sin(self.freqs * time)

        return signal + sine


class SignalRandAddSquarePulse(RandomizableTransform):
    """
    Add a random square pulse signal to the input signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        v: Optional[float] = 1.0,
        boundaries: Tuple[float, float] = Tuple[None, None],
        frequencies: Tuple[float, float] = Tuple[None, None],
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            v: scaling factor
            boundaries: list defining lower and upper boundaries for the signal drop, lower and upper values need to be positive example : ``[0.2, 0.6]``
            frequencies: list defining lower and upper frequencies for sinusoidal signal generation example : ``[0.001, 0.02]``
        """
        super().__init__()
        if boundaries is None or None in boundaries:
            raise ValueError("Incompatible values: boundaries needs to be a list of float.")
        if (boundaries is None) or (None in boundaries) or (any(x < 0 for x in boundaries)):
            raise ValueError("Incompatible values: boundaries needs to be a list of positive float.")

        self.v = v
        self.boundaries = boundaries
        self.frequencies = frequencies

    def _randomize(self):
        super().randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        self._randomize()
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)
        length = signal.shape[1]

        time = np.arange(0, length, 1)
        squaredpulse = self.magnitude * square(self.freqs * time)

        return signal + squaredpulse


class SignalRandAddGaussianNoise(RandomizableTransform):
    """
    Add a random sinusoidal signal to the input signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self, v: Optional[float] = 1.0, boundaries: Tuple[float, float] = Tuple[None, None], *args, **kwargs
    ) -> None:
        """
        Args:
            v: scaling factor
            boundaries: list defining lower and upper boundaries for the signal drop, lower and upper values need to be positive example : ``[0.2, 0.6]``
            frequencies: list defining lower and upper frequencies for sinusoidal signal generation example : ``[0.001, 0.02]``
        """
        super().__init__()
        if boundaries is None or None in boundaries:
            raise ValueError("Incompatible values: boundaries needs to be a list of float.")
        if (boundaries is None) or (None in boundaries) or (any(x < 0 for x in boundaries)):
            raise ValueError("Incompatible values: boundaries needs to be a list of positive float.")

        self.v = v
        self.boundaries = boundaries

    def _randomize(self):
        super().randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        self._randomize()
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)
        length = signal.shape[1]

        time = np.arange(0, length, 1)
        gaussiannoise = self.magnitude * np.random.normal(size=length)

        return signal + gaussiannoise


class SignalRandAddSinePartial(RandomizableTransform):
    """
    Randomly drop a portion of a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        v: Optional[float] = 1.0,
        boundaries: Tuple[float, float] = Tuple[None, None],
        frequencies: Tuple[float, float] = Tuple[None, None],
        fraction: Tuple[float, float] = Tuple[None, None],
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            v: scaling factor
            boundaries: list defining lower and upper boundaries for the signal drop, lower and upper values need to be positive example : ``[0.2, 0.6]``
        """
        super().__init__()
        if boundaries is None or None in boundaries:
            raise ValueError("Incompatible values: boundaries needs to be a list of float.")
        if (boundaries is None) or (None in boundaries) or (any(x < 0 for x in boundaries)):
            raise ValueError("Incompatible values: boundaries needs to be a list of positive float.")
        self.v = v
        self.boundaries = boundaries
        self.frequencies = frequencies
        self.fraction = fraction

    def _randomize(self):
        super().randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.fracs = self.R.uniform(low=self.fraction[0], high=self.fraction[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

    def _paste_slices(self, tup):
        pos, w, max_w = tup
        max_w = max_w.shape[len(max_w.shape) - 1]
        wall_min = max(pos, 0)
        wall_max = min(pos + w, max_w)
        block_min = -min(pos, 0)
        block_max = max_w - max(pos + w, max_w)
        block_max = block_max if block_max != 0 else None
        return slice(wall_min, wall_max), slice(block_min, block_max)

    def _paste(self, wall, block, loc):
        loc_zip = zip(loc, block.shape, wall)
        wall_slices, block_slices = zip(*map(self._paste_slices, loc_zip))

        wall[:, wall_slices[0]] = block[block_slices[0]]

        if wall.shape[0] == 1:
            wall = wall.squeeze()
        return wall

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        self._randomize()
        if len(signal.shape) > 1:
            length = signal.shape[len(signal.shape) - 1]
        else:
            signal = signal[np.newaxis, :]
            length = signal.shape[1]

        time_partial = np.arange(0, round(self.fracs * length), 1)
        sine_partial = self.v * self.magnitude * np.sin(self.freqs * time_partial)
        loc = np.random.choice(range(length))
        signal = self._paste(signal, sine_partial, (loc,))

        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)

        return signal


class SignalRandAddSquarePulsePartial(RandomizableTransform):
    """
    Randomly drop a portion of a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        v: Optional[float] = 1.0,
        boundaries: Tuple[float, float] = Tuple[None, None],
        frequencies: Tuple[float, float] = Tuple[None, None],
        fraction: Tuple[float, float] = Tuple[None, None],
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            v: scaling factor
            boundaries: list defining lower and upper boundaries for the signal drop, lower and upper values need to be positive example : ``[0.2, 0.6]``
        """
        super().__init__()
        if boundaries is None or None in boundaries:
            raise ValueError("Incompatible values: boundaries needs to be a list of float.")
        if (boundaries is None) or (None in boundaries) or (any(x < 0 for x in boundaries)):
            raise ValueError("Incompatible values: boundaries needs to be a list of positive float.")
        self.v = v
        self.boundaries = boundaries
        self.frequencies = frequencies
        self.fraction = fraction

    def _randomize(self):
        super().randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])
        self.fracs = self.R.uniform(low=self.fraction[0], high=self.fraction[1])
        self.freqs = self.R.uniform(low=self.frequencies[0], high=self.frequencies[1])

    def _paste_slices(self, tup):
        pos, w, max_w = tup
        max_w = max_w.shape[len(max_w.shape) - 1]
        wall_min = max(pos, 0)
        wall_max = min(pos + w, max_w)
        block_min = -min(pos, 0)
        block_max = max_w - max(pos + w, max_w)
        block_max = block_max if block_max != 0 else None
        return slice(wall_min, wall_max), slice(block_min, block_max)

    def _paste(self, wall, block, loc):
        loc_zip = zip(loc, block.shape, wall)
        wall_slices, block_slices = zip(*map(self._paste_slices, loc_zip))

        wall[:, wall_slices[0]] = block[block_slices[0]]

        if wall.shape[0] == 1:
            wall = wall.squeeze()
        return wall

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        self._randomize()
        if len(signal.shape) > 1:
            length = signal.shape[len(signal.shape) - 1]
        else:
            signal = signal[np.newaxis, :]
            length = signal.shape[1]

        time_partial = np.arange(0, round(self.fracs * length), 1)
        squaredpulse_partial = self.v * self.magnitude * square(self.freqs * time_partial)

        loc = np.random.choice(range(length))
        signal = self._paste(signal, squaredpulse_partial, (loc,))

        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)

        return signal


class SignalNormalize(Transform):
    """
    Resample signal to the target sampling rate.
    """

    backend = [TransformBackends.NUMPY]

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: signal to be normalized
        """
        if len(signal.shape) > 1:
            inputs_channels = signal.shape[0]
        else:
            inputs_channels = 1
            signal = np.expand_dims(signal, axis=0)

        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        return signal


class SignalStandardize(Transform):
    """
    Standardize a signal.
    """

    backend = [TransformBackends.NUMPY]

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: signal to be standardized
        """
        if len(signal.shape) > 1:
            inputs_channels = signal.shape[0]
        else:
            inputs_channels = 1
            signal = np.expand_dims(signal, axis=0)

        signal = (signal - signal.mean()) / signal.std()
        return signal


class SignalZeroPad(Transform):
    """
    Standardize a signal.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, targetlength: int = None, *args, **kwargs) -> None:
        """
        Args:
            targetlength: target length of the signal
        """
        super().__init__()
        self.targetlength = targetlength

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: signal to be padded
        """

        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)

        extended_signal = np.zeros([signal.shape[0], self.targetlength])
        siglength = np.min([self.targetlength, signal.shape[1]])
        extended_signal[:, :siglength] = signal[:, :siglength]

        return extended_signal


class SignalFillEmpty(Transform):
    """
    Standardize a signal.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, replacement: float = None, *args, **kwargs) -> None:
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
    Remove a frequency from a signal.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self, frequency: float = None, quality_factor: float = None, sampling_freq: float = None, *args, **kwargs
    ) -> None:
        """
        Args:
            frequency: frequency value to be removed from the signal
            quality_factor:
            sampling_freq:
        """
        super().__init__()
        self.frequency = frequency
        self.quality_factor = quality_factor
        self.sampling_freq = sampling_freq

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: signal to be filtered
        """

        b_notch, a_notch = iirnotch(self.frequency, self.quality_factor, self.sampling_freq)
        y_notched = filtfilt(b_notch, a_notch, signal)

        if len(y_notched.shape) == 1:
            y_notched = np.expand_dims(y_notched, axis=0)

        return y_notched


class SignalShortTimeFourier(Transform):
    """
    Generate short time Fourier transform of a signal.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, frequency: int = None, nperseg: int = None, noverlap: int = None, *args, **kwargs) -> None:
        """
        Args:
            frequency: signal frequency
            nperseg:
            noverlap:
        """
        super().__init__()
        self.frequency = frequency
        self.nperseg = nperseg
        self.noverlap = noverlap

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: signal to be filtered
        """
        f, t, Sxx = spectrogram(signal, fs=self.frequency, nperseg=self.nperseg, noverlap=self.noverlap)

        if len(signal.shape) > 1:
            Sxx = np.transpose(Sxx, [0, 2, 1])
        else:
            Sxx = np.transpose(Sxx, [1, 0])

        Sxx = np.abs(Sxx)
        mask = Sxx > 0
        Sxx[mask] = np.log(Sxx[mask])

        Sxx = (Sxx - np.mean(Sxx)) / np.std(Sxx)

        sx_norm = np.transpose(Sxx)

        if len(signal.shape) == 1:
            sx_norm = np.expand_dims(sx_norm, axis=0)
        else:
            sx_norm = np.transpose(sx_norm, [2, 0, 1])

        return sx_norm


class SignalContinousWavelet(Transform):
    """
    Generate continuous wavelet transform of a signal.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, type: str = None, length: int = None, frequency: int = None, *args, **kwargs) -> None:
        """
        Args:
            type: mother wavelet type.
            length:
            frequency:
        """
        super().__init__()
        self.frequency = frequency
        self.length = length
        self.type = type

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: signal for which to compute continuous wavelet transform
        """
        mother_wavelet = self.type
        spread = np.arange(1, self.length + 1, 1)
        scales = central_frequency(mother_wavelet) * self.frequency / spread

        coeffs, frequency = cwt(signal, scales, mother_wavelet, 1.0 / self.frequency)

        if len(signal.shape) == 1:
            coeffs = np.expand_dims(coeffs, axis=0)
        else:
            coeffs = np.transpose(coeffs, [1, 0, 2])

        return coeffs
