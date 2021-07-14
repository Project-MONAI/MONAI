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

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from monai.config import IgniteInfo
from monai.transforms import apply_transform
from monai.utils import min_version, optional_import
from monai.utils.enums import CommonKeys

if TYPE_CHECKING:
    from ignite.engine import EventEnum
else:
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

__all__ = [
    "IterationEvents",
    "GanKeys",
    "get_devices_spec",
    "default_prepare_batch",
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


class GanKeys:
    """
    A set of common keys for generative adversarial networks.

    """

    REALS = "reals"
    FAKES = "fakes"
    LATENTS = "latents"
    GLOSS = "g_loss"
    DLOSS = "d_loss"


def get_devices_spec(devices: Optional[Sequence[torch.device]] = None) -> List[torch.device]:
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

    return devices


def default_prepare_batch(
    batchdata: Dict[str, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    """
    Default function to prepare the data for current iteration.
    Refer to ignite: https://pytorch.org/ignite/v0.4.5/generated/ignite.engine.create_supervised_trainer.html
    #ignite.engine.create_supervised_trainer.

    Returns:
        image, label(optional).

    """
    if not isinstance(batchdata, dict):
        raise AssertionError("default prepare_batch expects dictionary input data.")
    if isinstance(batchdata.get(CommonKeys.LABEL), torch.Tensor):
        return (
            batchdata[CommonKeys.IMAGE].to(device=device, non_blocking=non_blocking),
            batchdata[CommonKeys.LABEL].to(device=device, non_blocking=non_blocking),
        )
    if GanKeys.REALS in batchdata:
        return batchdata[GanKeys.REALS].to(device=device, non_blocking=non_blocking)
    return batchdata[CommonKeys.IMAGE].to(device=device, non_blocking=non_blocking), None


def default_make_latent(
    num_latents: int,
    latent_size: int,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    return torch.randn(num_latents, latent_size).to(device=device, non_blocking=non_blocking)


def engine_apply_transform(batch: Any, output: Any, transform: Callable[..., Dict]):
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
