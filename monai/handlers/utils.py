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

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Sequence, Union

import numpy as np
import torch

from monai.utils import ensure_tuple, exact_version, optional_import

idist, _ = optional_import("ignite", "0.4.2", exact_version, "distributed")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")

__all__ = [
    "stopping_fn_from_metric",
    "stopping_fn_from_loss",
    "evenly_divisible_all_gather",
    "write_per_image_metric",
    "write_metric_summary",
]


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


def evenly_divisible_all_gather(data: torch.Tensor) -> torch.Tensor:
    """
    Utility function for distributed data parallel to pad at first dim to make it evenly divisible and all_gather.

    Args:
        data: source tensor to pad and execute all_gather in distributed data parallel.

    """
    if not torch.is_tensor(data):
        raise ValueError("input data must be PyTorch Tensor.")

    if idist.get_world_size() <= 1:
        return data

    # make sure the data is evenly-divisible on multi-GPUs
    length = data.shape[0]
    all_lens = idist.all_gather(length)
    max_len = max(all_lens).item()
    if length < max_len:
        size = [max_len - length] + list(data.shape[1:])
        data = torch.cat([data, data.new_full(size, 0)], dim=0)
    # all gather across all processes
    data = idist.all_gather(data)
    # delete the padding NaN items
    return torch.cat([data[i * max_len : i * max_len + l, ...] for i, l in enumerate(all_lens)], dim=0)


def write_per_image_metric(
    class_labels: Sequence[str],
    images: Sequence[str],
    metric: np.ndarray,
    filepath: str,
    deli: str = "\t",
):
    """
    Utility function to write the raw metric data of every image into a file, every line is for 1 image.
    The input metric data metric must have at least 2 dims(batch, classes).

    Args:
        class_labels: label string for every class in the metric data.
        images: name or path of every image corresponding to the metric data.
        metric: raw metric data for all images, it must have at least 2 dims(batch, classes).
        filepath: target file path to save the result, for example: "/workspace/data/mean_dice_raw.csv".
        deli: the delimiter charactor in the file, default to "\t".

    """

    if metric.ndim < 2:
        raise ValueError("metric must have at least 2 dims(batch, classes).")

    if not (isinstance(filepath, str) and filepath[-4:] == ".csv"):
        raise AssertionError("filepath must be a string with CSV format.")

    with open(filepath, "w") as f:
        f.write(f"filename{deli}{deli.join(class_labels)}\n")
        for i, image in enumerate(metric):
            f.write(f"{images[i]}{deli}{deli.join([str(c) for c in image])}\n")


def write_metric_summary(
    class_labels: Sequence[str],
    metric: np.ndarray,
    filepath: str,
    summary_ops: Union[str, Sequence[str]],
    deli: str = "\t",
):
    """
    Utility function to compute summary report of metric data on all the images, every line is for 1 class.

    Args:
        class_labels: label string for every class in the metric data.
        metric: raw metric data for all images, it must have at least 2 dims(batch, classes).
        filepath: target file path to save the result, for example: "/workspace/data/mean_dice_summary.csv".
        summary_ops: expected computation operations to generate the summary report.
            it can be: "*" or list of strings.
            "*" - generate summary report with all the supported operations.
            list of strings - generate summary report with specified operations, they should be within this list:
            [`mean`, `median`, `max`, `min`, `90percent`, `std`].
        deli: the delimiter charactor in the file, default to "\t".

    """

    if metric.ndim < 2:
        raise ValueError("metric must have at least 2 dims(batch, classes).")

    if not (isinstance(filepath, str) and filepath[-4:] == ".csv"):
        raise AssertionError("filepath must be a string with CSV format.")

    supported_ops = OrderedDict(
        {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "max": np.nanmax,
            "min": np.nanmin,
            "90percent": lambda x: np.nanpercentile(x, 10),
            "std": np.nanstd,
        }
    )
    ops = ensure_tuple(summary_ops)
    if "*" in ops:
        ops = tuple(supported_ops.keys())

    with open(filepath, "w") as f:
        f.write(f"class{deli}{deli.join(ops)}\n")
        for i, c in enumerate(metric.transpose()):
            f.write(f"{class_labels[i]}{deli}{deli.join([f'{supported_ops[k](c):.4f}' for k in ops])}\n")
