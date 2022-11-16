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

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from monai.inferers import Inferer
from monai.inferers.utils import compute_importance_map, sliding_window_inference
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple, look_up_option

__all__ = ["SlidingWindowHoVerNetInferer"]


class SlidingWindowHoVerNetInferer(Inferer):
    """
    Sliding window method for HoVerNet model inference,
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
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a tqdm progress bar.
        cache_roi_weight_map: whether to pre-compute the ROI weight map.
        cpu_thresh: when provided, dynamically switch to stitching on cpu (to save gpu memory)
            when input image volume is larger than this threshold (in pixels/voxels).
            Otherwise use ``"device"``. Thus, the output may end-up on either cpu or gpu.
        extra_input_padding: the amount of padding for the input image, which is a tuple of even number of pads.
            Refer to to the `pad` argument of `torch.nn.functional.pad` for more details.
        pad_output: wether to pad the inference output to match window size

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
        sw_device: Optional[Union[torch.device, str]] = None,
        device: Optional[Union[torch.device, str]] = None,
        progress: bool = False,
        cache_roi_weight_map: bool = False,
        cpu_thresh: Optional[int] = None,
        extra_input_padding: Optional[Tuple[int]] = None,
        pad_output: bool = False,
    ) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device
        self.progress = progress
        self.cpu_thresh = cpu_thresh
        self.extra_input_padding = extra_input_padding
        self.pad_output = pad_output

        # compute_importance_map takes long time when computing on cpu. We thus
        # compute it once if it's static and then save it for future usage
        self.roi_weight_map = None
        try:
            if cache_roi_weight_map and isinstance(roi_size, Sequence) and min(roi_size) > 0:  # non-dynamic roi size
                if device is None:
                    device = "cpu"
                self.roi_weight_map = compute_importance_map(
                    ensure_tuple(self.roi_size), mode=mode, sigma_scale=sigma_scale, device=device
                )
            if cache_roi_weight_map and self.roi_weight_map is None:
                warnings.warn("cache_roi_weight_map=True, but cache is not created. (dynamic roi_size?)")
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller roi_size, or use mode='constant' instead of mode='gaussian'. "
            ) from e

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """

        device = self.device
        if device is None and self.cpu_thresh is not None and inputs.shape[2:].numel() > self.cpu_thresh:
            device = "cpu"  # stitch in cpu memory if image is too large

        if self.extra_input_padding:
            image_size_original = inputs.shape[2:]
            num_spatial_dims = len(image_size_original)
            inputs = F.pad(
                inputs,
                pad=tuple(self.extra_input_padding),
                mode=look_up_option(self.padding_mode, PytorchPadMode),
                value=self.cval,
            )

        results = sliding_window_inference(
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
            device,
            self.progress,
            self.roi_weight_map,
            self.pad_output,
            *args,
            **kwargs,
        )

        if self.extra_input_padding:
            extra_slicing: List[slice] = []
            for sp in range(len(self.extra_input_padding) // 2):
                slice_dim = slice(
                    self.extra_input_padding[sp * 2],
                    image_size_original[num_spatial_dims - sp - 1] + self.extra_input_padding[sp * 2],
                )
                extra_slicing.insert(0, slice_dim)
            for k, v in results.items():
                results[k] = v[extra_slicing]

        return results
