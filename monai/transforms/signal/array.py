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

import numpy as np
from typing import Sequence, Optional, Union, Any
from monai.transforms.transform import Transform
from monai.utils import optional_import
from monai.utils.enums import TransformBackends

zoom, has_zoom = optional_import("scipy.ndimage", name="zoom")
resample_poly, has_resample_poly = optional_import("scipy.signal", name="resample_poly")
fft, has_resample_fft = optional_import("scipy.signal", name="resample")
shift, has_shift = optional_import("scipy.ndimage.interpolation.shift", name="shift")


__all__ = ["SignalResample"]


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


class SignalRandShift(Transform):
    """
    Apply a random shift on a signal
    """
    backend = [TransformBackends.NUMPY]
    
    def __init__(
        self, 
        magnitude: Sequence[float]= [-1.0,1.0], 
        mode: str='wrap',
        filling: Optional[Union[np.ndarray,Any]] = 0.0,
        v: float = 1.0
    ) -> None:
        """
        Args:
            magnitude: magnitude of the signal shift, taken randomly in the defined range. Defaults to ``[-1.0, 1.0]``
            filling: value to fill past edges of input if mode is ‘constant’. Default is 0.0. see for mode details
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
            mode: define how the extension of the input array is done beyond its boundaries, see for more details 
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
            v: scaling factor
        """    
        self.magnitude = magnitude
        self.filling = filling
        self.mode = mode
        self.v = v
        
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Args:
            signal: input 1 dimension signal to be resampled
        """
        if len(signal.shape) == 1:
            signal = np.expand_dims(signal, axis=0)  
        length = signal.shape[1]
        factor = self.v * np.random.uniform(low=self.magnitude[0],
                                    high=self.magnitude[1])
        shift_idx = round(factor * length)
        inputs = shift(input=inputs,
                       mode=self.mode,
                       shift=shift_idx, 
                       cval=self.filling)
        
        return signal
