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

idist, _ = optional_import("ignite", "0.4.2", exact_version, "distributed")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")

__all__ = ["stopping_fn_from_metric", "stopping_fn_from_loss", "evenly_divisible_all_gather"]


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


def evenly_divisible_all_gather(data: torch.Tensor, pad_dim: int = 0):
    """
    Utility function for distributed data parallel to pad tensor to make it evenly divisible for all_gather.

    Args:
        data: source tensor to pad and execute all_gather in distributed data parallel.
        pad_dim: which dimension to pad NaN data to make it evenly divisible, default to dim 0.

    """
    if idist.get_world_size() <= 1:
        return data
    # make sure the data is evenly-divisible on multi-GPUs
    length = data.shape[pad_dim]
    all_lens = idist.all_gather(length)
    max_len = max(all_lens).item()
    if length < max_len:
        size = [max_len - length] + list(data.shape[1:])
        data = torch.cat([data, data.new_full(size, float("NaN"))], dim=0)
    # all gather across all processes
    data = idist.all_gather(data)
    # delete the padding NaN items
    return torch.cat([data[i * max_len : i * max_len + l, ...] for i, l in enumerate(all_lens)], dim=0)
