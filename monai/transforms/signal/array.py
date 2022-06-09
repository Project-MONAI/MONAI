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

__all__ = ["SignalRandDrop",
           "SignalRandScale",
           "SignalRandShift",
           "SignalResample",
           "RandAddSine1d",
           "RandAddSquarePulse1d",
           "RandAddGaussianNoise1d"]

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

        if len(signal.shape) > 1:
            signal = np.squeeze(signal)

        return signal


class SignalRandShift(RandomizableTransform):
    """
    Apply a random shift on a signal
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        mode: Optional[str] = "wrap",
        filling: Optional[float] = 0.0,
        v: Optional[float] = 1.0,
        boundaries: Tuple[float, float] = Tuple[None, None],
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            mode: define how the extension of the input array is done beyond its boundaries, see for more details :
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html.
            filling: value to fill past edges of input if mode is ‘constant’. Default is 0.0. see for mode details :
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html.
            v: scaling factor
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

        if len(signal.shape) > 1:
            signal = np.squeeze(signal)

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
        return factor * signal


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

        return signal

class SignalRandAddSine(RandomizableTransform):
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

        return signal
