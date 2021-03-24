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

import logging
from typing import TYPE_CHECKING, Callable, Optional

from monai.data import CSVSaver
from monai.handlers.utils import evenly_divisible_all_gather, string_list_all_gather
from monai.utils import ImageMetaKey as Key
from monai.utils import exact_version, optional_import

idist, _ = optional_import("ignite", "0.4.4", exact_version, "distributed")
Events, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import("ignite.engine", "0.4.4", exact_version, "Engine")


class ClassificationSaver:
    """
    Event handler triggered on completing every iteration to save the classification predictions as CSV file.
    If running in distributed data parallel, only saves CSV file in the specified rank.

    """

    def __init__(
        self,
        output_dir: str = "./",
        filename: str = "predictions.csv",
        overwrite: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        name: Optional[str] = None,
        save_rank: int = 0,
    ) -> None:
        """
        Args:
            output_dir: output CSV file directory.
            filename: name of the saved CSV file name.
            overwrite: whether to overwriting existing CSV file content. If we are not overwriting,
                then we check if the results have been previously saved, and load them to the prediction_dict.
            batch_transform: a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
            output_transform: a callable that is used to transform the
                ignite.engine.output into the form expected model prediction data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
            name: identifier of logging.logger to use, defaulting to `engine.logger`.
            save_rank: only the handler on specified rank will save to CSV file in multi-gpus validation,
                default to 0.

        """
        self._expected_rank: bool = idist.get_rank() == save_rank
        self.saver = CSVSaver(output_dir, filename, overwrite)
        self.batch_transform = batch_transform
        self.output_transform = output_transform

        self.logger = logging.getLogger(name)
        self._name = name

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        if self._expected_rank and not engine.has_event_handler(self.saver.finalize, Events.COMPLETED):
            engine.add_event_handler(Events.COMPLETED, lambda engine: self.saver.finalize())

    def __call__(self, engine: Engine) -> None:
        """
        This method assumes self.batch_transform will extract metadata from the input batch.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        _meta_data = self.batch_transform(engine.state.batch)
        if Key.FILENAME_OR_OBJ in _meta_data:
            # all gather filenames across ranks, only filenames are necessary
            _meta_data = {Key.FILENAME_OR_OBJ: string_list_all_gather(_meta_data[Key.FILENAME_OR_OBJ])}
        # all gather predictions across ranks
        _engine_output = evenly_divisible_all_gather(self.output_transform(engine.state.output))

        if self._expected_rank:
            self.saver.save_batch(_engine_output, _meta_data)
