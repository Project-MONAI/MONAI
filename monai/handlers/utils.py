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

from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist

from monai.utils import exact_version, optional_import

if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")

__all__ = ["stopping_fn_from_metric", "stopping_fn_from_loss", "all_gather"]


def stopping_fn_from_metric(metric_name: str) -> Callable[[Engine], Any]:
    """
    Returns a stopping function for ignite.handlers.EarlyStopping using the given metric name.
    """

    def stopping_fn(engine: Engine):
        return engine.state.metrics[metric_name]

    return stopping_fn


def stopping_fn_from_loss() -> Callable[[Engine], Any]:
    """
    Returns a stopping function for ignite.handlers.EarlyStopping using the loss value.
    """

    def stopping_fn(engine: Engine):
        return -engine.state.output

    return stopping_fn


def all_gather(tensor):
    """
    All gather the data of tensor value in distributed data parallel.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("should not execute all_gather operation before torch.distributed is ready.")
    # create placeholder to collect the data from all processes
    output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    return torch.cat(output, dim=0)
