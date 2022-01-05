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

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import IgniteInfo, KeysCollection, PathLike
from monai.utils import ensure_tuple, look_up_option, min_version, optional_import

idist, _ = optional_import("ignite", IgniteInfo.OPT_IMPORT_VERSION, min_version, "distributed")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")

__all__ = ["stopping_fn_from_metric", "stopping_fn_from_loss", "write_metrics_reports", "from_engine"]


def stopping_fn_from_metric(metric_name: str):
    """
    Returns a stopping function for ignite.handlers.EarlyStopping using the given metric name.
    """

    def stopping_fn(engine: Engine):
        return engine.state.metrics[metric_name]

    return stopping_fn


def stopping_fn_from_loss():
    """
    Returns a stopping function for ignite.handlers.EarlyStopping using the loss value.
    """

    def stopping_fn(engine: Engine):
        return -engine.state.output  # type:ignore

    return stopping_fn


def write_metrics_reports(
    save_dir: PathLike,
    images: Optional[Sequence[str]],
    metrics: Optional[Dict[str, Union[torch.Tensor, np.ndarray]]],
    metric_details: Optional[Dict[str, Union[torch.Tensor, np.ndarray]]],
    summary_ops: Optional[Union[str, Sequence[str]]],
    deli: str = "\t",
    output_type: str = "csv",
):
    """
    Utility function to write the metrics into files, contains 3 parts:
    1. if `metrics` dict is not None, write overall metrics into file, every line is a metric name and value pair.
    2. if `metric_details` dict is not None,  write raw metric data of every image into file, every line for 1 image.
    3. if `summary_ops` is not None, compute summary based on operations on `metric_details` and write to file.

    Args:
        save_dir: directory to save all the metrics reports.
        images: name or path of every input image corresponding to the metric_details data.
            if None, will use index number as the filename of every input image.
        metrics: a dictionary of (metric name, metric value) pairs.
        metric_details: a dictionary of (metric name, metric raw values) pairs, usually, it comes from metrics
            computation, for example, the raw value can be the mean_dice of every channel of every input image.
        summary_ops: expected computation operations to generate the summary report.
            it can be: None, "*" or list of strings, default to None.
            None - don't generate summary report for every expected metric_details.
            "*" - generate summary report for every metric_details with all the supported operations.
            list of strings - generate summary report for every metric_details with specified operations, they
            should be within list: ["mean", "median", "max", "min", "<int>percentile", "std", "notnans"].
            the number in "<int>percentile" should be [0, 100], like: "15percentile". default: "90percentile".
            for more details, please check: https://numpy.org/doc/stable/reference/generated/numpy.nanpercentile.html.
            note that: for the overall summary, it computes `nanmean` of all classes for each image first,
            then compute summary. example of the generated summary report::

                class    mean    median    max    5percentile 95percentile  notnans
                class0  6.0000   6.0000   7.0000   5.1000      6.9000       2.0000
                class1  6.0000   6.0000   6.0000   6.0000      6.0000       1.0000
                mean    6.2500   6.2500   7.0000   5.5750      6.9250       2.0000

        deli: the delimiter character in the file, default to "\t".
        output_type: expected output file type, supported types: ["csv"], default to "csv".

    """
    if output_type.lower() != "csv":
        raise ValueError(f"unsupported output type: {output_type}.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if metrics is not None and len(metrics) > 0:
        with open(os.path.join(save_dir, "metrics.csv"), "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}{deli}{str(v)}\n")
    if metric_details is not None and len(metric_details) > 0:
        for k, v in metric_details.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if v.ndim == 0:
                # reshape to [1, 1] if no batch and class dims
                v = v.reshape((1, 1))
            elif v.ndim == 1:
                # reshape to [N, 1] if no class dim
                v = v.reshape((-1, 1))

            # add the average value of all classes to v
            class_labels = ["class" + str(i) for i in range(v.shape[1])] + ["mean"]
            v = np.concatenate([v, np.nanmean(v, axis=1, keepdims=True)], axis=1)

            with open(os.path.join(save_dir, f"{k}_raw.csv"), "w") as f:
                f.write(f"filename{deli}{deli.join(class_labels)}\n")
                for i, b in enumerate(v):
                    f.write(f"{images[i] if images is not None else str(i)}{deli}{deli.join([str(c) for c in b])}\n")

            if summary_ops is not None:
                supported_ops = OrderedDict(
                    {
                        "mean": np.nanmean,
                        "median": np.nanmedian,
                        "max": np.nanmax,
                        "min": np.nanmin,
                        "90percentile": lambda x: np.nanpercentile(x[0], x[1]),
                        "std": np.nanstd,
                        "notnans": lambda x: (~np.isnan(x)).sum(),
                    }
                )
                ops = ensure_tuple(summary_ops)
                if "*" in ops:
                    ops = tuple(supported_ops.keys())

                def _compute_op(op: str, d: np.ndarray):
                    if not op.endswith("percentile"):
                        c_op = look_up_option(op, supported_ops)
                        return c_op(d)

                    threshold = int(op.split("percentile")[0])
                    return supported_ops["90percentile"]((d, threshold))  # type: ignore

                with open(os.path.join(save_dir, f"{k}_summary.csv"), "w") as f:
                    f.write(f"class{deli}{deli.join(ops)}\n")
                    for i, c in enumerate(np.transpose(v)):
                        f.write(f"{class_labels[i]}{deli}{deli.join([f'{_compute_op(k, c):.4f}' for k in ops])}\n")


def from_engine(keys: KeysCollection, first: bool = False):
    """
    Utility function to simplify the `batch_transform` or `output_transform` args of ignite components
    when handling dictionary or list of dictionaries(for example: `engine.state.batch` or `engine.state.output`).
    Users only need to set the expected keys, then it will return a callable function to extract data from
    dictionary and construct a tuple respectively.

    If data is a list of dictionaries after decollating, extract expected keys and construct lists respectively,
    for example, if data is `[{"A": 1, "B": 2}, {"A": 3, "B": 4}]`, from_engine(["A", "B"]): `([1, 3], [2, 4])`.

    It can help avoid a complicated `lambda` function and make the arg of metrics more straight-forward.
    For example, set the first key as the prediction and the second key as label to get the expected data
    from `engine.state.output` for a metric::

        from monai.handlers import MeanDice, from_engine

        metric = MeanDice(
            include_background=False,
            output_transform=from_engine(["pred", "label"])
        )

    Args:
        keys: specified keys to extract data from dictionary or decollated list of dictionaries.
        first: whether only extract specified keys from the first item if input data is a list of dictionaries,
            it's used to extract the scalar data which doesn't have batch dim and was replicated into every
            dictionary when decollating, like `loss`, etc.


    """
    keys = ensure_tuple(keys)

    def _wrapper(data):
        if isinstance(data, dict):
            return tuple(data[k] for k in keys)
        if isinstance(data, list) and isinstance(data[0], dict):
            # if data is a list of dictionaries, extract expected keys and construct lists,
            # if `first=True`, only extract keys from the first item of the list
            ret = [data[0][k] if first else [i[k] for i in data] for k in keys]
            return tuple(ret) if len(ret) > 1 else ret[0]

    return _wrapper
