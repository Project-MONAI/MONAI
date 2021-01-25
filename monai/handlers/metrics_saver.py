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

from typing import TYPE_CHECKING, Sequence, Optional, Callable, Union
import os
import torch

from monai.utils import exact_version, optional_import, ensure_tuple
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
        metric_details: expected metric details to save into files, for example: mean dice
            of every channel of every image in the validation dataset.
            the data in `engine.state.details` must contain 2 dims: (batch, classes).
            it can be: None, "*" or list of strings.
            None - don't save any metrics into files.
            "*" - save all the existing metrics in `engine.state.metric_details` dict into separate files.
            list of strings - specify the expected metrics to save.
        batch_transform: callable function to extract the meta_dict from input batch data if saving metric details.
            used to extract filenames from input dict data.
        compute_summary: whether to compute a summary report against all the images.
        save_rank: only the handler on specified rank will save to files in multi-gpus validation, default to 0.

    """

    def __init__(
        self,
        save_dir: str,
        metrics: Optional[Union[str, Sequence[str]]] = None,
        metric_details: Optional[Union[str, Sequence[str]]] = None,
        batch_transform: Callable = lambda x: x,
        compute_summary: bool = False,
        save_rank: int = 0,
    ) -> None:
        self.save_dir = save_dir
        self.metrics = ensure_tuple(metrics) if metrics is not None else None
        self.metric_details = ensure_tuple(metric_details) if metric_details is not None else None
        self.batch_transform = batch_transform
        self.compute_summary = compute_summary
        self.save_rank = save_rank
        self._filenames = None

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        engine.add_event_handler(Events.STARTED, self._started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._get_filenames)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self)

    def _started(self, engine: Engine) -> None:
        self._filenames = list()

    def _get_filenames(self, engine: Engine) -> None:
        if self.metric_details is not None:
            self._filenames += self.batch_transform(engine.state.batch)["filename_or_obj"]

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
                            f.write(k + "\t" + str(v) + "\n")

        if self.metric_details is not None and hasattr(engine.state, "metric_details") \
                and len(engine.state.metric_details) > 0:
            _filenames = "\t".join(self._filenames)

            if ws > 1:
                if get_torch_version_tuple() > (1, 6, 0):
                    # all gather across all processes
                    _filenames = "\t".join(idist.all_gather(_filenames))
                else:
                    raise RuntimeError(
                        "MetricsSaver can not save metric details in distributed mode with PyTorch < 1.7.0."
                    )
            if idist.get_rank() == self.save_rank:
                _filenames = _filenames.split("\t")
                for k, v in engine.state.metric_details.items():
                    if k in self.metric_details or "*" in self.metric_details:
                        nans = torch.isnan(v)
                        not_nans = (~nans).float()
                        average = list()
                        with open(os.path.join(self.save_dir, k + "_raw.csv"), "w") as f:
                            classes = "\t".join(["class" + str(i) for i in range(v.shape[1])])
                            f.write("filename\t" + classes + "\taverage\n")
                            for i, image in enumerate(v):
                                ave = image.nansum() / not_nans[i].sum()
                                average.append(ave)
                                classes = "\t".join([str(i.item()) for i in image])
                                f.write(_filenames[i] + "\t" + classes + "\t" + str(ave.item()) + "\n")
