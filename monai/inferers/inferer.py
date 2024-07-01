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

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pydoc import locate
from typing import Any

import torch
import torch.nn as nn

from monai.apps.utils import get_logger
from monai.data.meta_tensor import MetaTensor
from monai.data.thread_buffer import ThreadBuffer
from monai.inferers.merger import AvgMerger, Merger
from monai.inferers.splitter import Splitter
from monai.inferers.utils import compute_importance_map, sliding_window_inference
from monai.utils import BlendMode, PatchKeys, PytorchPadMode, ensure_tuple, optional_import
from monai.visualize import CAM, GradCAM, GradCAMpp

logger = get_logger(__name__)

__all__ = [
    "Inferer",
    "PatchInferer",
    "SimpleInferer",
    "SlidingWindowInferer",
    "SaliencyInferer",
    "SliceInferer",
    "SlidingWindowInfererAdapt",
]


class Inferer(ABC):
    """
    A base class for model inference.
    Extend this class to support operations during inference, e.g. a sliding window method.

    Example code::

        device = torch.device("cuda:0")
        transform = Compose([ToTensor(), LoadImage(image_only=True)])
        data = transform(img_path).to(device)
        model = UNet(...).to(device)
        inferer = SlidingWindowInferer(...)

        model.eval()
        with torch.no_grad():
            pred = inferer(inputs=data, network=model)
        ...

    """

    @abstractmethod
    def __call__(self, inputs: torch.Tensor, network: Callable, *args: Any, **kwargs: Any) -> Any:
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


class PatchInferer(Inferer):
    """
    Inference on patches instead of the whole image based on Splitter and Merger.
    This splits the input image into patches and then merge the resulted patches.

    Args:
        splitter: a `Splitter` object that split the inputs into patches. Defaults to None.
            If not provided or None, the inputs are considered to be already split into patches.
            In this case, the output `merged_shape` and the optional `cropped_shape` cannot be inferred
            and should be explicitly provided.
        merger_cls: a `Merger` subclass that can be instantiated to merges patch outputs.
            It can also be a string that matches the name of a class inherited from `Merger` class.
            Defaults to `AvgMerger`.
        batch_size: batch size for patches. If the input tensor is already batched [BxCxWxH],
            this adds additional batching [(Bp*B)xCxWpxHp] for inference on patches.
            Defaults to 1.
        preprocessing: a callable that process patches before the being fed to the network.
            Defaults to None.
        postprocessing: a callable that process the output of the network.
            Defaults to None.
        output_keys: if the network output is a dictionary, this defines the keys of
            the output dictionary to be used for merging.
            Defaults to None, where all the keys are used.
        match_spatial_shape: whether to crop the output to match the input shape. Defaults to True.
        buffer_size: number of patches to be held in the buffer with a separate thread for batch sampling. Defaults to 0.
        merger_kwargs: arguments to be passed to `merger_cls` for instantiation.
            `merged_shape` is calculated automatically based on the input shape and
            the output patch shape unless it is passed here.
    """

    def __init__(
        self,
        splitter: Splitter | None = None,
        merger_cls: type[Merger] | str = AvgMerger,
        batch_size: int = 1,
        preprocessing: Callable | None = None,
        postprocessing: Callable | None = None,
        output_keys: Sequence | None = None,
        match_spatial_shape: bool = True,
        buffer_size: int = 0,
        **merger_kwargs: Any,
    ) -> None:
        Inferer.__init__(self)
        # splitter
        if not isinstance(splitter, (Splitter, type(None))):
            if not isinstance(splitter, Splitter):
                raise TypeError(
                    f"'splitter' should be a `Splitter` object that returns: "
                    "an iterable of pairs of (patch, location) or a MetaTensor that has `PatchKeys.LOCATION` metadata)."
                    f"{type(splitter)} is given."
                )
        self.splitter = splitter

        # merger
        if isinstance(merger_cls, str):
            valid_merger_cls: type[Merger]
            # search amongst implemented mergers in MONAI
            valid_merger_cls, merger_found = optional_import("monai.inferers.merger", name=merger_cls)
            if not merger_found:
                # try to locate the requested merger class (with dotted path)
                valid_merger_cls = locate(merger_cls)  # type: ignore
            if valid_merger_cls is None:
                raise ValueError(f"The requested `merger_cls` ['{merger_cls}'] does not exist.")
            merger_cls = valid_merger_cls
        if not issubclass(merger_cls, Merger):
            raise TypeError(f"'merger' should be a subclass of `Merger`, {merger_cls} is given.")
        self.merger_cls = merger_cls
        self.merger_kwargs = merger_kwargs

        # pre-processor (process patch before the network)
        if preprocessing is not None and not callable(preprocessing):
            raise TypeError(f"'preprocessing' should be a callable object, {type(preprocessing)} is given.")
        self.preprocessing = preprocessing

        # post-processor (process the output of the network)
        if postprocessing is not None and not callable(postprocessing):
            raise TypeError(f"'postprocessing' should be a callable object, {type(postprocessing)} is given.")
        self.postprocessing = postprocessing

        # batch size for patches
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be a positive number, {batch_size} is given.")
        self.batch_size = batch_size

        # model output keys
        self.output_keys = output_keys

        # whether to crop the output to match the input shape
        self.match_spatial_shape = match_spatial_shape

        # buffer size for multithreaded batch sampling
        self.buffer_size = buffer_size

    def _batch_sampler(
        self, patches: Iterable[tuple[torch.Tensor, Sequence[int]]] | MetaTensor
    ) -> Iterator[tuple[torch.Tensor, Sequence, int]]:
        """Generate batch of patches and locations

        Args:
            patches: a tensor or list of tensors

        Yields:
            A batch of patches (torch.Tensor or MetaTensor), a sequence of location tuples, and the batch size
        """
        if isinstance(patches, MetaTensor):
            total_size = len(patches)
            for i in range(0, total_size, self.batch_size):
                batch_size = min(self.batch_size, total_size - i)
                yield patches[i : i + batch_size], patches[i : i + batch_size].meta[PatchKeys.LOCATION], batch_size  # type: ignore
        else:
            buffer: Iterable | ThreadBuffer
            if self.buffer_size > 0:
                # Use multi-threading to sample patches with a buffer
                buffer = ThreadBuffer(patches, buffer_size=self.buffer_size, timeout=0.1)
            else:
                buffer = patches
            patch_batch: list[Any] = [None] * self.batch_size
            location_batch: list[Any] = [None] * self.batch_size
            idx_in_batch = 0
            for sample in buffer:
                patch_batch[idx_in_batch] = sample[0]
                location_batch[idx_in_batch] = sample[1]
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    # concatenate batch of patches to create a tensor
                    yield torch.cat(patch_batch), location_batch, idx_in_batch
                    patch_batch = [None] * self.batch_size
                    location_batch = [None] * self.batch_size
                    idx_in_batch = 0
            if idx_in_batch > 0:
                # concatenate batch of patches to create a tensor
                yield torch.cat(patch_batch[:idx_in_batch]), location_batch, idx_in_batch

    def _ensure_tuple_outputs(self, outputs: Any) -> tuple:
        if isinstance(outputs, dict):
            if self.output_keys is None:
                self.output_keys = list(outputs.keys())  # model's output keys
            return tuple(outputs[k] for k in self.output_keys)
        return ensure_tuple(outputs, wrap_array=True)

    def _run_inference(self, network: Callable, patch: torch.Tensor, *args: Any, **kwargs: Any) -> tuple:
        # pre-process
        if self.preprocessing:
            patch = self.preprocessing(patch)
        # inference
        outputs = network(patch, *args, **kwargs)
        # post-process
        if self.postprocessing:
            outputs = self.postprocessing(outputs)
        # ensure we have a tuple of model outputs to support multiple outputs
        return self._ensure_tuple_outputs(outputs)

    def _initialize_mergers(self, inputs, outputs, patches, batch_size):
        in_patch = torch.chunk(patches, batch_size)[0]
        mergers = []
        ratios = []
        for out_patch_batch in outputs:
            out_patch = torch.chunk(out_patch_batch, batch_size)[0]
            # calculate the ratio of input and output patch sizes
            ratio = tuple(op / ip for ip, op in zip(in_patch.shape[2:], out_patch.shape[2:]))

            # calculate merged_shape and cropped_shape
            merger_kwargs = self.merger_kwargs.copy()
            cropped_shape, merged_shape = self._get_merged_shapes(inputs, out_patch, ratio)
            if "merged_shape" not in merger_kwargs:
                merger_kwargs["merged_shape"] = merged_shape
                if merger_kwargs["merged_shape"] is None:
                    raise ValueError("`merged_shape` cannot be `None`.")
            if "cropped_shape" not in merger_kwargs:
                merger_kwargs["cropped_shape"] = cropped_shape

            # initialize the merger
            merger = self.merger_cls(**merger_kwargs)

            # store mergers and input/output ratios
            mergers.append(merger)
            ratios.append(ratio)

        return mergers, ratios

    def _aggregate(self, outputs, locations, batch_size, mergers, ratios):
        for output_patches, merger, ratio in zip(outputs, mergers, ratios):
            # split batched output into individual patches and then aggregate
            for in_loc, out_patch in zip(locations, torch.chunk(output_patches, batch_size)):
                out_loc = [round(l * r) for l, r in zip(in_loc, ratio)]
                merger.aggregate(out_patch, out_loc)

    def _get_merged_shapes(self, inputs, out_patch, ratio):
        """Define the shape of merged tensors (non-padded and padded)"""
        if self.splitter is None:
            return None, None

        # input spatial shapes
        original_spatial_shape = self.splitter.get_input_shape(inputs)
        padded_spatial_shape = self.splitter.get_padded_shape(inputs)

        # output spatial shapes
        output_spatial_shape = tuple(round(s * r) for s, r in zip(original_spatial_shape, ratio))
        padded_output_spatial_shape = tuple(round(s * r) for s, r in zip(padded_spatial_shape, ratio))

        # output shapes
        cropped_shape = out_patch.shape[:2] + output_spatial_shape
        merged_shape = out_patch.shape[:2] + padded_output_spatial_shape

        if not self.match_spatial_shape:
            cropped_shape = merged_shape

        return cropped_shape, merged_shape

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Args:
            inputs: input data for inference, a torch.Tensor, representing an image or batch of images.
                However if the data is already split, it can be fed by providing a list of tuple (patch, location),
                or a MetaTensor that has metadata for `PatchKeys.LOCATION`. In both cases no splitter should be provided.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        patches_locations: Iterable[tuple[torch.Tensor, Sequence[int]]] | MetaTensor
        if self.splitter is None:
            # handle situations where the splitter is not provided
            if isinstance(inputs, torch.Tensor):
                if isinstance(inputs, MetaTensor):
                    if PatchKeys.LOCATION not in inputs.meta:
                        raise ValueError(
                            "`PatchKey.LOCATION` does not exists in `inputs.meta`. "
                            "If the inputs are already split into patches, the location of patches needs to be "
                            "provided as `PatchKey.LOCATION` metadata in a MetaTensor. "
                            "If the input is not already split, please provide `splitter`."
                        )
                else:
                    raise ValueError(
                        "`splitter` should be set if the input is not already split into patches. "
                        "For inputs that are split, the location of patches needs to be provided as "
                        "(image, location) pairs, or as `PatchKey.LOCATION` metadata in a MetaTensor. "
                        f"The provided inputs type is {type(inputs)}."
                    )
            patches_locations = inputs
        else:
            # apply splitter
            patches_locations = self.splitter(inputs)

        ratios: list[float] = []
        mergers: list[Merger] = []
        for patches, locations, batch_size in self._batch_sampler(patches_locations):
            # run inference
            outputs = self._run_inference(network, patches, *args, **kwargs)
            # initialize the mergers
            if not mergers:
                mergers, ratios = self._initialize_mergers(inputs, outputs, patches, batch_size)
            # aggregate outputs
            self._aggregate(outputs, locations, batch_size, mergers, ratios)

        # finalize the mergers and get the results
        merged_outputs = [merger.finalize() for merger in mergers]

        # return according to the model output
        if self.output_keys:
            return dict(zip(self.output_keys, merged_outputs))
        if len(merged_outputs) == 1:
            return merged_outputs[0]
        return merged_outputs


class SimpleInferer(Inferer):
    """
    SimpleInferer is the normal inference method that run model forward() directly.
    Usage example can be found in the :py:class:`monai.inferers.Inferer` base class.

    """

    def __init__(self) -> None:
        Inferer.__init__(self)

    def __call__(
        self, inputs: torch.Tensor, network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
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
        overlap: Amount of overlap between scans along each spatial dimension, defaults to ``0.25``.
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
        cache_roi_weight_map: whether to precompute the ROI weight map.
        cpu_thresh: when provided, dynamically switch to stitching on cpu (to save gpu memory)
            when input image volume is larger than this threshold (in pixels/voxels).
            Otherwise use ``"device"``. Thus, the output may end-up on either cpu or gpu.
        buffer_steps: the number of sliding window iterations along the ``buffer_dim``
            to be buffered on ``sw_device`` before writing to ``device``.
            (Typically, ``sw_device`` is ``cuda`` and ``device`` is ``cpu``.)
            default is None, no buffering. For the buffer dim, when spatial size is divisible by buffer_steps*roi_size,
            (i.e. no overlapping among the buffers) non_blocking copy may be automatically enabled for efficiency.
        buffer_dim: the spatial dimension along which the buffers are created.
            0 indicates the first spatial dimension. Default is -1, the last spatial dimension.
        with_coord: whether to pass the window coordinates to ``network``. Defaults to False.
            If True, the ``network``'s 2nd input argument should accept the window coordinates.

    Note:
        ``sw_batch_size`` denotes the max number of windows per network inference iteration,
        not the batch size of inputs.

    """

    def __init__(
        self,
        roi_size: Sequence[int] | int,
        sw_batch_size: int = 1,
        overlap: Sequence[float] | float = 0.25,
        mode: BlendMode | str = BlendMode.CONSTANT,
        sigma_scale: Sequence[float] | float = 0.125,
        padding_mode: PytorchPadMode | str = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: torch.device | str | None = None,
        device: torch.device | str | None = None,
        progress: bool = False,
        cache_roi_weight_map: bool = False,
        cpu_thresh: int | None = None,
        buffer_steps: int | None = None,
        buffer_dim: int = -1,
        with_coord: bool = False,
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
        self.buffer_steps = buffer_steps
        self.buffer_dim = buffer_dim
        self.with_coord = with_coord

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
                f"roi size {self.roi_size}, mode={mode}, sigma_scale={sigma_scale}, device={device}\n"
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """

        device = kwargs.pop("device", self.device)
        buffer_steps = kwargs.pop("buffer_steps", self.buffer_steps)
        buffer_dim = kwargs.pop("buffer_dim", self.buffer_dim)

        if device is None and self.cpu_thresh is not None and inputs.shape[2:].numel() > self.cpu_thresh:
            device = "cpu"  # stitch in cpu memory if image is too large

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
            device,
            self.progress,
            self.roi_weight_map,
            None,
            buffer_steps,
            buffer_dim,
            self.with_coord,
            *args,
            **kwargs,
        )


class SlidingWindowInfererAdapt(SlidingWindowInferer):
    """
    SlidingWindowInfererAdapt extends SlidingWindowInferer to automatically switch to buffered and then to CPU stitching,
    when OOM on GPU. It also records a size of such large images to automatically
    try CPU stitching for the next large image of a similar size.  If the stitching 'device' input parameter is provided,
    automatic adaptation won't be attempted, please keep the default option device = None for adaptive behavior.
    Note: the output might be on CPU (even if the input was on GPU), if the GPU memory was not sufficient.

    """

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """

        # if device is provided, use without any adaptations
        if self.device is not None:
            return super().__call__(inputs, network, *args, **kwargs)

        skip_buffer = self.buffer_steps is not None and self.buffer_steps <= 0
        cpu_cond = self.cpu_thresh is not None and inputs.shape[2:].numel() > self.cpu_thresh
        gpu_stitching = inputs.is_cuda and not cpu_cond
        buffered_stitching = inputs.is_cuda and cpu_cond and not skip_buffer
        buffer_steps = max(1, self.buffer_steps) if self.buffer_steps is not None else 1
        buffer_dim = -1

        sh = list(inputs.shape[2:])
        max_dim = sh.index(max(sh))
        if inputs.shape[max_dim + 2] / inputs.shape[-1] >= 2:
            buffer_dim = max_dim

        for _ in range(10):  # at most 10 trials
            try:
                return super().__call__(
                    inputs,
                    network,
                    *args,
                    device=inputs.device if gpu_stitching else torch.device("cpu"),
                    buffer_steps=buffer_steps if buffered_stitching else None,
                    buffer_dim=buffer_dim,
                    **kwargs,
                )
            except RuntimeError as e:
                if not gpu_stitching and not buffered_stitching or "OutOfMemoryError" not in str(type(e).__name__):
                    raise e

                logger.info(e)

                if gpu_stitching:  # if failed on gpu
                    gpu_stitching = False
                    self.cpu_thresh = inputs.shape[2:].numel() - 1  # update thresh

                    if skip_buffer:
                        buffered_stitching = False
                        logger.warning(f"GPU stitching failed, attempting on CPU, image dim {inputs.shape}.")

                    else:
                        buffered_stitching = True
                        self.buffer_steps = buffer_steps
                        logger.warning(
                            f"GPU stitching failed, buffer {buffer_steps} dim {buffer_dim}, image dim {inputs.shape}."
                        )
                elif buffer_steps > 1:
                    buffer_steps = max(1, buffer_steps // 2)
                    self.buffer_steps = buffer_steps
                    logger.warning(
                        f"GPU buffered stitching failed, image dim {inputs.shape} reducing buffer to {buffer_steps}."
                    )
                else:
                    buffered_stitching = False
                    logger.warning(f"GPU buffered stitching failed, attempting on CPU, image dim {inputs.shape}.")
        raise RuntimeError(  # not possible to finish after the trials
            f"SlidingWindowInfererAdapt {skip_buffer} {cpu_cond} {gpu_stitching} {buffered_stitching} {buffer_steps}"
        )


class SaliencyInferer(Inferer):
    """
    SaliencyInferer is inference with activation maps.

    Args:
        cam_name: expected CAM method name, should be: "CAM", "GradCAM" or "GradCAMpp".
        target_layers: name of the model layer to generate the feature map.
        class_idx: index of the class to be visualized. if None, default to argmax(logits).
        args: other optional args to be passed to the `__init__` of cam.
        kwargs: other optional keyword args to be passed to `__init__` of cam.

    """

    def __init__(
        self, cam_name: str, target_layers: str, class_idx: int | None = None, *args: Any, **kwargs: Any
    ) -> None:
        Inferer.__init__(self)
        if cam_name.lower() not in ("cam", "gradcam", "gradcampp"):
            raise ValueError("cam_name should be: 'CAM', 'GradCAM' or 'GradCAMpp'.")
        self.cam_name = cam_name.lower()
        self.target_layers = target_layers
        self.class_idx = class_idx
        self.args = args
        self.kwargs = kwargs

    def __call__(self, inputs: torch.Tensor, network: nn.Module, *args: Any, **kwargs: Any):  # type: ignore
        """Unified callable function API of Inferers.

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: other optional args to be passed to the `__call__` of cam.
            kwargs: other optional keyword args to be passed to `__call__` of cam.

        """
        cam: CAM | GradCAM | GradCAMpp
        if self.cam_name == "cam":
            cam = CAM(network, self.target_layers, *self.args, **self.kwargs)
        elif self.cam_name == "gradcam":
            cam = GradCAM(network, self.target_layers, *self.args, **self.kwargs)
        else:
            cam = GradCAMpp(network, self.target_layers, *self.args, **self.kwargs)

        return cam(inputs, self.class_idx, *args, **kwargs)


class SliceInferer(SlidingWindowInferer):
    """
    SliceInferer extends SlidingWindowInferer to provide slice-by-slice (2D) inference when provided a 3D volume.
    A typical use case could be a 2D model (like 2D segmentation UNet) operates on the slices from a 3D volume,
    and the output is a 3D volume with 2D slices aggregated. Example::

        # sliding over the `spatial_dim`
        inferer = SliceInferer(roi_size=(64, 256), sw_batch_size=1, spatial_dim=1)
        output = inferer(input_volume, net)

    Args:
        spatial_dim: Spatial dimension over which the slice-by-slice inference runs on the 3D volume.
            For example ``0`` could slide over axial slices. ``1`` over coronal slices and ``2`` over sagittal slices.
        args: other optional args to be passed to the `__init__` of base class SlidingWindowInferer.
        kwargs: other optional keyword args to be passed to `__init__` of base class SlidingWindowInferer.

    Note:
        ``roi_size`` in SliceInferer is expected to be a 2D tuple when a 3D volume is provided. This allows
        sliding across slices along the 3D volume using a selected ``spatial_dim``.

    """

    def __init__(self, spatial_dim: int = 0, *args: Any, **kwargs: Any) -> None:
        self.spatial_dim = spatial_dim
        super().__init__(*args, **kwargs)
        self.orig_roi_size = ensure_tuple(self.roi_size)

    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """
        Args:
            inputs: 3D input for inference
            network: 2D model to execute inference on slices in the 3D input
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.
        """
        if self.spatial_dim > 2:
            raise ValueError("`spatial_dim` can only be `0, 1, 2` with `[H, W, D]` respectively.")

        # Check if ``roi_size`` tuple is 2D and ``inputs`` tensor is 3D
        self.roi_size = ensure_tuple(self.roi_size)
        if len(self.orig_roi_size) == 2 and len(inputs.shape[2:]) == 3:
            self.roi_size = list(self.orig_roi_size)
            self.roi_size.insert(self.spatial_dim, 1)
        else:
            raise RuntimeError(
                f"Currently, only 2D `roi_size` ({self.orig_roi_size}) with 3D `inputs` tensor (shape={inputs.shape}) is supported."
            )

        return super().__call__(inputs=inputs, network=lambda x: self.network_wrapper(network, x, *args, **kwargs))

    def network_wrapper(
        self,
        network: Callable[..., torch.Tensor | Sequence[torch.Tensor] | dict[Any, torch.Tensor]],
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | dict[Any, torch.Tensor]:
        """
        Wrapper handles inference for 2D models over 3D volume inputs.
        """
        #  Pass 4D input [N, C, H, W]/[N, C, D, W]/[N, C, D, H] to the model as it is 2D.
        x = x.squeeze(dim=self.spatial_dim + 2)
        out = network(x, *args, **kwargs)

        #  Unsqueeze the network output so it is [N, C, D, H, W] as expected by
        # the default SlidingWindowInferer class
        if isinstance(out, torch.Tensor):
            return out.unsqueeze(dim=self.spatial_dim + 2)

        if isinstance(out, Mapping):
            for k in out.keys():
                out[k] = out[k].unsqueeze(dim=self.spatial_dim + 2)
            return out

        return tuple(out_i.unsqueeze(dim=self.spatial_dim + 2) for out_i in out)
