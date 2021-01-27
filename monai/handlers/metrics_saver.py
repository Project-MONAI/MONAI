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

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Union

import numpy as np
import torch

from monai.utils import ensure_tuple, exact_version, optional_import
from monai.utils.module import get_torch_version_tuple

Events, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Events")
idist, _ = optional_import("ignite", "0.4.2", exact_version, "distributed")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.2", exact_version, "Engine")


class MetricsSaver:
    """
    ignite handler to save metrics values and details into expected files.

    Args:
        save_dir: directory to save the metrics and metric details.
        metrics: expected final metrics to save into files, can be: None, "*" or list of strings.
            None - don't save any metrics into files.
            "*" - save all the existing metrics in `engine.state.metrics` dict into separate files.
            list of strings - specify the expected metrics to save.
            default to "*" to save all the metrics into `metrics.csv`.
        metric_details: expected metric details to save into files, for example: mean dice
            of every channel of every image in the validation dataset.
            the data in `engine.state.metric_details` must contain at least 2 dims: (batch, classes, ...),
            if not, will unsequeeze to 2 dims.
            this arg can be: None, "*" or list of strings.
            None - don't save any metrics into files.
            "*" - save all the existing metrics in `engine.state.metric_details` dict into separate files.
            list of strings - specify the expected metrics to save.
            if not None, every metric will save a separate `{metric name}_raw.csv` file.
        batch_transform: callable function to extract the meta_dict from input batch data if saving metric details.
            used to extract filenames from input dict data.
        summary_ops: expected computation operations to generate the summary report.
            it can be: None, "*" or list of strings.
            None - don't generate summary report for every expected metric_details
            "*" - generate summary report for every metric_details with all the supported operations.
            list of strings - generate summary report for every metric_details with specified operations, they
            should be within this list: [`mean`, `median`, `max`, `min`, `90percent`, `std`].
            default to None.
        save_rank: only the handler on specified rank will save to files in multi-gpus validation, default to 0.
        delimiter: the delimiter charactor in CSV file, default to "\t".

    """

    def __init__(
        self,
        save_dir: str,
        metrics: Optional[Union[str, Sequence[str]]] = "*",
        metric_details: Optional[Union[str, Sequence[str]]] = None,
        batch_transform: Callable = lambda x: x,
        summary_ops: Optional[Union[str, Sequence[str]]] = None,
        save_rank: int = 0,
        delimiter: str = "\t",
    ) -> None:
        self.save_dir = save_dir
        self.metrics = ensure_tuple(metrics) if metrics is not None else None
        self.metric_details = ensure_tuple(metric_details) if metric_details is not None else None
        self.batch_transform = batch_transform
        self.summary_ops = ensure_tuple(summary_ops) if summary_ops is not None else None
        self.save_rank = save_rank
        self.deli = delimiter
        self._filenames: List[str] = []

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.STARTED, self._started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._get_filenames)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self)

    def _started(self, engine: Engine) -> None:
        self._filenames = []

    def _get_filenames(self, engine: Engine) -> None:
        if self.metric_details is not None:
            _filenames = list(ensure_tuple(self.batch_transform(engine.state.batch)["filename_or_obj"]))
            self._filenames += _filenames

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        ws = idist.get_world_size()
        if self.save_rank >= ws:
            raise ValueError("target rank is greater than the distributed group size.s")

        if self.metrics is not None and len(engine.state.metrics) > 0:
            if idist.get_rank() == self.save_rank:
                # only save metrics to file in specified rank
                with open(os.path.join(self.save_dir, "metrics.csv"), "w") as f:
                    for k, v in engine.state.metrics.items():
                        if k in self.metrics or "*" in self.metrics:
                            f.write(f"{k}{self.deli}{str(v)}\n")

        if (
            self.metric_details is not None
            and hasattr(engine.state, "metric_details")
            and len(engine.state.metric_details) > 0
        ):
            _filenames = self.deli.join(self._filenames)

            if ws > 1:
                if get_torch_version_tuple() > (1, 6, 0):
                    # all gather across all processes
                    _filenames = self.deli.join(idist.all_gather(_filenames))
                else:
                    raise RuntimeError(
                        "MetricsSaver can not save metric details in distributed mode with PyTorch < 1.7.0."
                    )
            if idist.get_rank() == self.save_rank:
                _files = _filenames.split(self.deli)
                for k, v in engine.state.metric_details.items():
                    if k in self.metric_details or "*" in self.metric_details:
                        if torch.is_tensor(v):
                            v = v.cpu().numpy()
                        if v.ndim == 0:
                            # reshape to [1, 1] if no batch and class dims
                            v = v.reshape((1, 1))
                        elif v.ndim == 1:
                            # reshape to [N, 1] if no class dim
                            v = v.reshape((-1, 1))

                        # add the average value to v
                        v = np.concatenate([v, np.nanmean(v, axis=1, keepdims=True)], axis=1)
                        with open(os.path.join(self.save_dir, k + "_raw.csv"), "w") as f:
                            class_labels = self.deli.join(["class" + str(i) for i in range(v.shape[1] - 1)])
                            row_labels = class_labels + f"{self.deli}average"
                            f.write(f"filename{self.deli}{row_labels}\n")
                            for i, image in enumerate(v):
                                classes = self.deli.join([str(d) for d in image])
                                f.write(f"{_files[i]}{self.deli}{classes}\n")

                        if self.summary_ops is not None:
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
                            ops = supported_ops.keys() if "*" in self.summary_ops else self.summary_ops

                            col_labels = row_labels.split(self.deli)
                            with open(os.path.join(self.save_dir, k + "_summary.csv"), "w") as f:
                                f.write(f"class{self.deli}{self.deli.join(ops)}\n")
                                for i, d in enumerate(v.transpose()):
                                    ops_labels = self.deli.join([f"{supported_ops[k](d):.4f}" for k in ops])
                                    f.write(f"{col_labels[i]}{self.deli}{ops_labels}\n")
