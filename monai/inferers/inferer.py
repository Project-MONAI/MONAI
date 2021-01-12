# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, Union

import torch

from monai.inferers.utils import sliding_window_inference
from monai.utils import BlendMode, PytorchPadMode

__all__ = ["Inferer", "SimpleInferer", "SlidingWindowInferer"]


class Inferer(ABC):
    """
    A base class for model inference.
    Extend this class to support operations during inference, e.g. a sliding window method.

    Example code::

        device = torch.device("cuda:0")
        data = ToTensor()(LoadImage()(filename=img_path)).to(device)
        model = UNet(...).to(device)
        inferer = SlidingWindowInferer(...)

        model.eval()
        with torch.no_grad():
            pred = inferer(inputs=data, network=model)
        ...

    """

    @abstractmethod
    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ):
        """
        Run inference on `inputs` with the `network` model.

        Args:
            inputs: input of the model inference.
            network: model for inference.
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class SimpleInferer(Inferer):
    """
    SimpleInferer is the normal inference method that run model forward() directly.
    Usage example can be found in the :py:class:`monai.inferers.Inferer` base class.

    """

    def __init__(self) -> None:
        Inferer.__init__(self)

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ):
        """Unified callable function API of Inferers.

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        return network(inputs, *args, **kwargs)


class SlidingWindowInferer(Inferer):
    """
    Sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().
    Usage example can be found in the :py:class:`monai.inferers.Inferer` base class.

    Args:
        roi_size: the window size to execute SlidingWindow evaluation.
            If it has non-positive components, the corresponding `inputs` size will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.

    Note:
        ``sw_batch_size`` denotes the max number of windows per network inference iteration,
        not the batch size of inputs.

    """

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
    ) -> None:
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        return sliding_window_inference(
            inputs,
            self.roi_size,
            self.sw_batch_size,
            network,
            self.overlap,
            self.mode,
            self.sigma_scale,
            self.padding_mode,
            self.cval,
            self.sw_device,
            self.device,
            *args,
            **kwargs,
        )
