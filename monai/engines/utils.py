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

from collections import UserDict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from monai.transforms import apply_transform
from monai.utils import exact_version, optional_import
from monai.utils.enums import CommonKeys

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum, State
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")
    EventEnum, _ = optional_import("ignite.engine", "0.4.4", exact_version, "EventEnum")
    State, _ = optional_import("ignite.engine", "0.4.4", exact_version, "State")

__all__ = [
    "get_devices_spec",
    "default_prepare_batch",
    "default_make_latent",
    "IterationEvents",
    "attach_ignite_engine",
    "engine_apply_transform",
]


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

        if len(devices) == 0:
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
    Refer to ignite: https://github.com/pytorch/ignite/blob/v0.4.2/ignite/engine/__init__.py#L28.

    Returns:
        image, label(optional).

    """
    if not isinstance(batchdata, dict):
        raise AssertionError("default prepare_batch expects dictionary input data.")
    if isinstance(batchdata.get(CommonKeys.LABEL, None), torch.Tensor):
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


class IterationEvents(EventEnum):
    """
    Additional Events engine can register and trigger in the iteration process.
    Refer to the example in ignite: https://github.com/pytorch/ignite/blob/master/ignite/engine/events.py#L146
    These Events can be triggered during training iteration:
    `FORWARD_COMPLETED` is the Event when `network(image, label)` completed.
    `LOSS_COMPLETED` is the Event when `loss(pred, label)` completed.
    `BACKWARD_COMPLETED` is the Event when `loss.backward()` completed.
    `MODEL_COMPLETED` is the Event when all the model related operations completed.

    """

    FORWARD_COMPLETED = "forward_completed"
    LOSS_COMPLETED = "loss_completed"
    BACKWARD_COMPLETED = "backward_completed"
    MODEL_COMPLETED = "model_completed"


class DictState(State):
    """
    Utility class that wrapper ignite State with python dict properties.

    """

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @staticmethod
    def from_state(state):
        dstate = DictState()
        dstate.__dict__ = state.__dict__
        return dstate


class EngineAsDict(UserDict):
    """
    Utility class that wrapper ignite Engine with python dict properties.

    """

    def __init__(self, engine: Engine):
        super().__init__()
        engine.state = DictState.from_state(engine.state)
        self.data = engine.__dict__


def attach_ignite_engine(engine: Engine, handler):
    """
    Attach MONAI handler to the specified ignite Engine.
    handler should be a monai.handlers.Handler.

    """
    from monai.handlers import Handler  # avoid circular import

    if not isinstance(handler, Handler):
        raise ValueError("handler must be monai.handlers.Handler.")
    dict_engine = EngineAsDict(engine)
    for event, func in handler.get_event_funcs().items():
        # pass the event as kwarg to handler callback
        engine.add_event_handler(event, func, data=dict_engine)


def engine_apply_transform(batch: Any, output: Any, transform: Callable):
    """
    Apply transform for the engine.state.batch and engine.state.output.
    If `batch` and `output` are dictionaries, temporarily combine them for the transform,
    otherwise, apply the transform for `output` data only.

    """
    if isinstance(batch, dict) and isinstance(output, dict):
        data = dict(batch)
        data.update(output)
        data = apply_transform(transform, data)
        for k, v in data.items():
            # split the output data of post transforms into `output` and `batch`,
            # `batch` should be read-only, so save the generated key-value into `output`
            if k in output or k not in batch:
                output[k] = v
            else:
                batch[k] = v
    else:
        output = apply_transform(transform, output)

    return batch, output
