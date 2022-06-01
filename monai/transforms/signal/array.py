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

from typing import Any, Optional, Union

import numpy as np

from monai.transforms.transform import RandomizableTransform, Transform
from monai.utils import optional_import
from monai.utils.enums import TransformBackends

zoom, has_zoom = optional_import("scipy.ndimage", name="zoom")
resample_poly, has_resample_poly = optional_import("scipy.signal", name="resample_poly")
fft, has_resample_fft = optional_import("scipy.signal", name="resample")
shift, has_shift = optional_import("scipy.ndimage.interpolation", name="shift")

assert has_zoom
assert has_resample_poly
assert has_resample_poly
assert has_resample_fft
assert has_shift

__all__ = ["SignalResample", "SignalRandShift"]


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
        mode: str = "wrap",
        filling: Optional[Union[np.ndarray, Any]] = 0.0,
        v: float = 1.0,
        boundaries: Optional[Union[np.ndarray, Any]] = None,
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
        super(SignalRandShift, self).__init__()

        self.filling = filling
        self.mode = mode
        self.v = v
        self.boundaries = boundaries

    def randomize(self):
        super().randomize(None)
        self.magnitude = self.R.uniform(low=self.boundaries[0], high=self.boundaries[1])

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        self.randomize()
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)
        length = signal.shape[1]
        factor = self.v * self.magnitude
        shift_idx = round(factor * length)
        signal = shift(input=signal, mode=self.mode, shift=shift_idx, cval=self.filling)

        if len(signal.shape) > 1:
            signal = np.squeeze(signal)

        return signal
