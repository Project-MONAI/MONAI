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

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from monai.transforms import apply_transform
from monai.utils import IgniteInfo, ensure_tuple, min_version, optional_import
from monai.utils.enums import CommonKeys, GanKeys

if TYPE_CHECKING:
    from ignite.engine import EventEnum
else:
    EventEnum, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum", as_type="base"
    )

__all__ = [
    "IterationEvents",
    "get_devices_spec",
    "default_prepare_batch",
    "PrepareBatch",
    "PrepareBatchDefault",
    "PrepareBatchExtraInput",
    "DiffusionPrepareBatch",
    "VPredictionPrepareBatch",
    "default_make_latent",
    "engine_apply_transform",
    "default_metric_cmp_fn",
]


class IterationEvents(EventEnum):
    """
    Additional Events engine can register and trigger in the iteration process.
    Refer to the example in ignite: https://pytorch.org/ignite/generated/ignite.engine.events.EventEnum.html.
    These Events can be triggered during training iteration:
    `FORWARD_COMPLETED` is the Event when `network(image, label)` completed.
    `LOSS_COMPLETED` is the Event when `loss(pred, label)` completed.
    `BACKWARD_COMPLETED` is the Event when `loss.backward()` completed.
    `MODEL_COMPLETED` is the Event when all the model related operations completed.
    `INNER_ITERATION_STARTED` is the Event when the iteration has an inner loop and the loop is started.
    `INNER_ITERATION_COMPLETED` is the Event when the iteration has an inner loop and the loop is completed.
    """

    FORWARD_COMPLETED = "forward_completed"
    LOSS_COMPLETED = "loss_completed"
    BACKWARD_COMPLETED = "backward_completed"
    MODEL_COMPLETED = "model_completed"
    INNER_ITERATION_STARTED = "inner_iteration_started"
    INNER_ITERATION_COMPLETED = "inner_iteration_completed"


def get_devices_spec(devices: Sequence[torch.device | str] | None = None) -> list[torch.device]:
    """
    Get a valid specification for one or more devices. If `devices` is None get devices for all CUDA devices available.
    If `devices` is and zero-length structure a single CPU compute device is returned. In any other cases `devices` is
    returned unchanged.

    Args:
        devices: list of devices to request, None for all GPU devices, [] for CPU.

    Raises:
        RuntimeError: When all GPUs are selected (``devices=None``) but no GPUs are available.

    Returns:
        list of torch.device: list of devices.

    """
    if devices is None:
        devices = [torch.device(f"cuda:{d:d}") for d in range(torch.cuda.device_count())]

        if not devices:
            raise RuntimeError("No GPU devices available.")

    elif len(devices) == 0:
        devices = [torch.device("cpu")]

    else:
        devices = list(devices)

    devices = [torch.device(d) if isinstance(d, str) else d for d in devices]
    return devices  # type: ignore


def default_prepare_batch(
    batchdata: dict[str, torch.Tensor] | torch.Tensor | Sequence[torch.Tensor],
    device: str | torch.device | None = None,
    non_blocking: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
    """
    Default function to prepare the data for current iteration.

    The input `batchdata` is either a single tensor, a pair of tensors, or a dictionary of data. In the first case the
    return value is the tensor and None, in the second case the return value is the two tensors, and in the dictionary
    case the return value depends on what keys are present. if `CommonKeys.IMAGE` and `CommonKeys.LABEL` are present
    then the tensors they key to are returned, if only `CommonKeys.IMAGE` is present that tensor and None is returned.
    If `CommonKeys.REALS` is present this is returned with None. All returned tensors are moved to the given device
    using the given non-blocking argument before being returned.

    This function implements the expected API for a `prepare_batch` callable in Ignite:
    https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.create_supervised_trainer.html

    Args:
        batchdata: input batch data which is either a single tensor, a pair, or a dictionary
        device: device to move every returned tensor to
        non_blocking: equivalent argument for `Tensor.to`
        kwargs: further arguments for `Tensor.to`

    Returns:
        image, label(optional).
    """
    if not isinstance(batchdata, dict):
        if isinstance(batchdata, torch.Tensor):
            return batchdata.to(device=device, non_blocking=non_blocking, **kwargs), None
        elif len(batchdata) == 2:
            image, label = batchdata
            return (
                image.to(device=device, non_blocking=non_blocking, **kwargs),
                label.to(device=device, non_blocking=non_blocking, **kwargs),
            )

        raise AssertionError("Default prepare_batch expects a single tensor, a tensor pair, or dictionary input data.")

    if isinstance(batchdata.get(CommonKeys.LABEL), torch.Tensor):
        return (
            batchdata[CommonKeys.IMAGE].to(device=device, non_blocking=non_blocking, **kwargs),
            batchdata[CommonKeys.LABEL].to(device=device, non_blocking=non_blocking, **kwargs),
        )

    if GanKeys.REALS in batchdata:
        return batchdata[GanKeys.REALS].to(device=device, non_blocking=non_blocking, **kwargs)

    return batchdata[CommonKeys.IMAGE].to(device=device, non_blocking=non_blocking, **kwargs), None


class PrepareBatch(ABC):
    """
    Interface of customized prepare_batch in the trainer or evaluator workflows.
    It takes the data of current batch, target device and non_blocking flag as input.
    Args `batchdata`, `device`, `non_blocking` refer to the ignite API:
    https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.create_supervised_trainer.html.
    `kwargs` supports other args for `Tensor.to()` API.
    """

    @abstractmethod
    def __call__(
        self,
        batchdata: dict[str, torch.Tensor],
        device: str | torch.device | None = None,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class PrepareBatchDefault(PrepareBatch):
    """
    This wraps `default_prepare_batch` to return `image` and `label` only, so is consistent with its API.
    """

    def __call__(
        self,
        batchdata: dict[str, torch.Tensor] | torch.Tensor | Sequence[torch.Tensor],
        device: str | torch.device | None = None,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
        """
        Args `batchdata`, `device`, `non_blocking` refer to the ignite API:
        https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.create_supervised_trainer.html.
        `kwargs` supports other args for `Tensor.to()` API.

        """
        return default_prepare_batch(batchdata, device, non_blocking, **kwargs)


class PrepareBatchExtraInput(PrepareBatch):
    """
    Customized prepare batch callable for trainers or evaluators which support extra input data for the network.
    Extra items are specified by the `extra_keys` parameter and are extracted from the input dictionary (ie. the batch).
    This uses `default_prepare_batch` but requires dictionary inputs.

    Args:
        extra_keys: If a string or sequence of strings is provided, values from the input dictionary are extracted from
            those keys and passed to the network as extra positional arguments. If a dictionary is provided, every pair
            `(k, v)` in that dictionary will become a new keyword argument assigning to `k` the value in the input
            dictionary keyed to `v`.
    """

    def __init__(self, extra_keys: str | Sequence[str] | dict[str, str]) -> None:
        self.extra_keys = extra_keys

    def __call__(
        self,
        batchdata: dict[str, torch.Tensor],
        device: str | torch.device | None = None,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple, dict]:
        """
        Args `batchdata`, `device`, `non_blocking` refer to the ignite API:
        https://pytorch.org/ignite/v0.4.8/generated/ignite.engine.create_supervised_trainer.html.
        `kwargs` supports other args for `Tensor.to()` API.
        """
        image, label = default_prepare_batch(batchdata, device, non_blocking, **kwargs)
        args_ = list()
        kwargs_ = dict()

        def _get_data(key: str) -> torch.Tensor:
            data = batchdata[key]

            if isinstance(data, torch.Tensor):
                data = data.to(device=device, non_blocking=non_blocking, **kwargs)

            return data

        if isinstance(self.extra_keys, (str, list, tuple)):
            for k in ensure_tuple(self.extra_keys):
                args_.append(_get_data(k))
        elif isinstance(self.extra_keys, dict):
            for k, v in self.extra_keys.items():
                kwargs_.update({k: _get_data(v)})

        return cast(torch.Tensor, image), cast(torch.Tensor, label), tuple(args_), kwargs_


class DiffusionPrepareBatch(PrepareBatch):
    """
    This class is used as a callable for the `prepare_batch` parameter of engine classes for diffusion training.

    Assuming a supervised training process, it will generate a noise field using `get_noise` for an input image, and
    return the image and noise field as the image/target pair plus the noise field the kwargs under the key "noise".
    This assumes the inferer being used in conjunction with this class expects a "noise" parameter to be provided.

    If the `condition_name` is provided, this must refer to a key in the input dictionary containing the condition
    field to be passed to the inferer. This will appear in the keyword arguments under the key "condition".

    """

    def __init__(self, num_train_timesteps: int, condition_name: str | None = None) -> None:
        self.condition_name = condition_name
        self.num_train_timesteps = num_train_timesteps

    def get_noise(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the noise tensor for input tensor `images`, override this for different noise distributions."""
        return torch.randn_like(images)

    def get_timesteps(self, images: torch.Tensor) -> torch.Tensor:
        """Get a timestep, by default this is a random integer between 0 and `self.num_train_timesteps`."""
        return torch.randint(0, self.num_train_timesteps, (images.shape[0],), device=images.device).long()

    def get_target(self, images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Return the target for the loss function, this is the `noise` value by default."""
        return noise

    def __call__(
        self,
        batchdata: dict[str, torch.Tensor],
        device: str | torch.device | None = None,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple, dict]:
        images, _ = default_prepare_batch(batchdata, device, non_blocking, **kwargs)
        noise = self.get_noise(images).to(device, non_blocking=non_blocking, **kwargs)
        timesteps = self.get_timesteps(images).to(device, non_blocking=non_blocking, **kwargs)

        target = self.get_target(images, noise, timesteps).to(device, non_blocking=non_blocking, **kwargs)
        infer_kwargs = {"noise": noise, "timesteps": timesteps}

        if self.condition_name is not None and isinstance(batchdata, Mapping):
            infer_kwargs["condition"] = batchdata[self.condition_name].to(device, non_blocking=non_blocking, **kwargs)

        # return input, target, arguments, and keyword arguments where noise is the target and also a keyword value
        return images, target, (), infer_kwargs


class VPredictionPrepareBatch(DiffusionPrepareBatch):
    """
    This class is used as a callable for the `prepare_batch` parameter of engine classes for diffusion training.

    Assuming a supervised training process, it will generate a noise field using `get_noise` for an input image, and
    from this compute the velocity using the provided scheduler. This value is used as the target in place of the
    noise field itself although the noise is field is in the kwargs under the key "noise". This assumes the inferer
    being used in conjunction with this class expects a "noise" parameter to be provided.

    If the `condition_name` is provided, this must refer to a key in the input dictionary containing the condition
    field to be passed to the inferer. This will appear in the keyword arguments under the key "condition".

    """

    def __init__(self, scheduler: nn.Module, num_train_timesteps: int, condition_name: str | None = None) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, condition_name=condition_name)
        self.scheduler = scheduler

    def get_target(self, images, noise, timesteps):
        return self.scheduler.get_velocity(images, noise, timesteps)  # type: ignore[operator]


def default_make_latent(
    num_latents: int,
    latent_size: int,
    device: str | torch.device | None = None,
    non_blocking: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    return torch.randn(num_latents, latent_size).to(device=device, non_blocking=non_blocking, **kwargs)


def engine_apply_transform(batch: Any, output: Any, transform: Callable[..., dict]) -> tuple[Any, Any]:
    """
    Apply transform on `batch` and `output`.
    If `batch` and `output` are dictionaries, temporarily combine them for the transform,
    otherwise, apply the transform for `output` data only.

    """
    if isinstance(batch, dict) and isinstance(output, dict):
        data = dict(batch)
        data.update(output)
        transformed_data = apply_transform(transform, data)

        if not isinstance(transformed_data, dict):
            raise AssertionError("With a dict supplied to apply_transform a single dict return is expected.")

        for k, v in transformed_data.items():
            # split the output data of post transforms into `output` and `batch`,
            # `batch` should be read-only, so save the generated key-value into `output`
            if k in output or k not in batch:
                output[k] = v
            else:
                batch[k] = v
    else:
        output = apply_transform(transform, output)

    return batch, output


def default_metric_cmp_fn(current_metric: float, prev_best: float) -> bool:
    """
    The default function to compare metric values between current metric and previous best metric.

    Args:
        current_metric: metric value of current round computation.
        prev_best: the best metric value of previous rounds to compare with.

    """
    return current_metric > prev_best
