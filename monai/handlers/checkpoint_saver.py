# Copyright 2020 MONAI Consortium
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
from typing import Optional

from monai.utils import exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Events")
Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")
ModelCheckpoint, _ = optional_import("ignite.handlers", "0.3.0", exact_version, "ModelCheckpoint")


class CheckpointSaver:
    """
    CheckpointSaver acts as an Ignite handler to save checkpoint data into files.
    It supports to save according to metrics result, epoch number, iteration number
    and last model or exception.

    Args:
        save_dir: the target directory to save the checkpoints.
        save_dict (dict): source objects that save to the checkpoint. examples::

            {'network': net, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

        name: identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
        file_prefix: prefix for the filenames to which objects will be saved.
        save_final: whether to save checkpoint or session at final iteration or exception.
        save_key_metric: whether to save checkpoint or session when the value of key_metric is
            higher than all the previous values during training.keep 4 decimal places of metric,
            checkpoint name is: {file_prefix}_key_metric=0.XXXX.pth.
        key_metric_name: the name of key_metric in ignite metrics dictionary.
            if None, use `engine.state.key_metric` instead.
        key_metric_n_saved: save top N checkpoints or sessions, sorted by the value of key
            metric in descending order.
        epoch_level: save checkpoint during training for every N epochs or every N iterations.
            `True` is epoch level, `False` is iteration level.
        save_interval: save checkpoint every N epochs, default is 0 to save no checkpoint.
        n_saved: save latest N checkpoints of epoch level or iteration level, 'None' is to save all.

    Note:
        CheckpointHandler can be used during training, validation or evaluation.
        example of saved files:

            - checkpoint_iteration=400.pth
            - checkpoint_iteration=800.pth
            - checkpoint_epoch=1.pth
            - checkpoint_final_iteration=1000.pth
            - checkpoint_key_metric=0.9387.pth

    """

    def __init__(
        self,
        save_dir: str,
        save_dict,
        name: Optional[str] = None,
        file_prefix: str = "",
        save_final: bool = False,
        save_key_metric: bool = False,
        key_metric_name: Optional[str] = None,
        key_metric_n_saved: int = 1,
        epoch_level: bool = True,
        save_interval: int = 0,
        n_saved: Optional[int] = None,
    ):
        assert save_dir is not None, "must provide directory to save the checkpoints."
        self.save_dir = save_dir
        assert save_dict is not None and len(save_dict) > 0, "must provide source objects to save."
        for k, v in save_dict.items():
            if hasattr(v, "module"):
                save_dict[k] = v.module
        self.save_dict = save_dict
        self.logger = logging.getLogger(name)
        self.epoch_level = epoch_level
        self.save_interval = save_interval
        self._final_checkpoint = self._key_metric_checkpoint = self._interval_checkpoint = None
        self._name = name

        if save_final:

            def _final_func(engine):
                return engine.state.iteration

            self._final_checkpoint = ModelCheckpoint(
                self.save_dir,
                file_prefix,
                score_function=_final_func,
                score_name="final_iteration",
                require_empty=False,
            )
        if save_key_metric:

            def _score_func(engine):
                if isinstance(key_metric_name, str):
                    metric_name = key_metric_name
                elif hasattr(engine.state, "key_metric_name") and isinstance(engine.state.key_metric_name, str):
                    metric_name = engine.state.key_metric_name
                else:
                    raise ValueError("must provde key_metric_name to save best validation model.")
                return round(engine.state.metrics[metric_name], 4)

            self._key_metric_checkpoint = ModelCheckpoint(
                self.save_dir,
                file_prefix,
                score_function=_score_func,
                score_name="key_metric",
                n_saved=key_metric_n_saved,
                require_empty=False,
            )
        if save_interval > 0:

            def _interval_func(engine):
                return engine.state.epoch if self.epoch_level else engine.state.iteration

            self._interval_checkpoint = ModelCheckpoint(
                self.save_dir,
                file_prefix,
                score_function=_interval_func,
                score_name="epoch" if self.epoch_level else "iteration",
                n_saved=n_saved,
                require_empty=False,
            )

    def attach(self, engine: Engine):
        if self._name is None:
            self.logger = engine.logger
        if self._final_checkpoint is not None:
            engine.add_event_handler(Events.COMPLETED, self.completed)
            engine.add_event_handler(Events.EXCEPTION_RAISED, self.exception_raised)
        if self._key_metric_checkpoint is not None:
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.metrics_completed)
        if self._interval_checkpoint is not None:
            if self.epoch_level:
                engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval), self.interval_completed)
            else:
                engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.save_interval), self.interval_completed)

    def completed(self, engine) -> None:
        """Callback for train or validation/evaluation completed Event.
        Save final checkpoint if configure save_final is True.

        """
        assert callable(self._final_checkpoint), "Error: _final_checkpoint function not specified."
        self._final_checkpoint(engine, self.save_dict)
        assert self.logger is not None
        assert hasattr(self.logger, "info"), "Error, provided logger has not info attribute."
        self.logger.info(f"Train completed, saved final checkpoint: {self._final_checkpoint.last_checkpoint}")

    def exception_raised(self, engine, e) -> None:
        """Callback for train or validation/evaluation exception raised Event.
        Save current data as final checkpoint if configure save_final is True.

        """
        assert callable(self._final_checkpoint), "Error: _final_checkpoint function not specified."
        self._final_checkpoint(engine, self.save_dict)
        assert self.logger is not None
        assert hasattr(self.logger, "info"), "Error, provided logger has not info attribute."
        self.logger.info(f"Exception_raised, saved exception checkpoint: {self._final_checkpoint.last_checkpoint}")

    def metrics_completed(self, engine) -> None:
        """Callback to compare metrics and save models in train or validation when epoch completed.

        """
        assert callable(self._key_metric_checkpoint), "Error: _key_metric_checkpoint function not specified."
        self._key_metric_checkpoint(engine, self.save_dict)

    def interval_completed(self, engine) -> None:
        """Callback for train epoch/iteration completed Event.
        Save checkpoint if configure save_interval = N

        """
        assert callable(self._interval_checkpoint), "Error: _interval_checkpoint function not specified."
        self._interval_checkpoint(engine, self.save_dict)
        assert self.logger is not None
        assert hasattr(self.logger, "info"), "Error, provided logger has not info attribute."
        if self.epoch_level:
            self.logger.info(f"Saved checkpoint at epoch: {engine.state.epoch}")
        else:
            self.logger.info(f"Saved checkpoint at iteration: {engine.state.iteration}")
