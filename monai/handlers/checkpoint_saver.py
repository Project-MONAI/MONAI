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
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint


class CheckpointSaver:
    """
    CheckpointSaver acts as an Ignite handler to save checkpoint data into files.
    It supports to save according to metrics result, epoch number, iteration number
    and last model or exception.

    Args:
        save_dir (str): the target directory to save the checkpoints.
        save_dict (dict): source objects that save to the checkpoint. examples::

            {'network': net, 'optimizer': optimizer, 'engine', engine}

        name (str): identifier of logging.logger to use, if None, defaulting to ``engine.logger``.
        file_prefix (str): prefix for the filenames to which objects will be saved.
        save_final (bool): whether to save checkpoint or session at final iteration or exception.
        save_key_metric (bool): whether to save checkpoint or session according to key metric value.
        key_metric_name (str): the name of key_metric in ignite metrics dictionary.
            if None, use `engine.state.key_metric` instead.
        key_metric_n_saved (int): save top N checkpoints or sessions sorted by the key metric.
        epoch_save_interval (int): save checkpoint or session every N epochs.
        epoch_n_saved (int): save latest N epochs' checkpoints or sessions, 'None' is to save all.
        iteration_save_interval (int): save checkpoint or session every N iterations.
        iteration_n_saved (int): save latest N iterations' checkpoints or sessions, 'None' is to save all.
        convert_to_single_gpu (bool): if current model is trained by multi GPU and
            want to save to single GPU model, need to save `object.module` instead.

    Note:
        CheckpointHandler can be used during training, validation or evaluation.
        example of saved files:
            checkpoint_iteration=400.pth
            checkpoint_iteration=800.pth
            checkpoint_epoch=1.pth
            checkpoint_final_iteration=1000.pth
            checkpoint_key_metric=0.9387.pth

    """
    def __init__(
        self,
        save_dir,
        save_dict,
        name=None,
        file_prefix='',
        save_final=False,
        save_key_metric=False,
        key_metric_name=None,
        key_metric_n_saved=1,
        epoch_save_interval=0,
        epoch_n_saved=None,
        iteration_save_interval=0,
        iteration_n_saved=None,
        convert_to_single_gpu=False
    ):
        assert save_dir is not None, "must provide directory to save the checkpoints."
        self.save_dir = save_dir
        assert save_dict is not None and len(save_dict) > 0, "must provide source objects to save."
        if convert_to_single_gpu:
            for k, v in save_dict.items():
                if hasattr(v, 'module'):
                    save_dict[k] = v.module
        self.save_dict = save_dict
        self.logger = None if name is None else logging.getLogger(name)
        self.save_final = save_final
        self.save_key_metric = save_key_metric
        self.key_metric_name = key_metric_name
        self.key_metric_n_saved = key_metric_n_saved
        self.epoch_save_interval = epoch_save_interval
        self.epoch_n_saved = epoch_n_saved
        self.iteration_save_interval = iteration_save_interval
        self.iteration_n_saved = iteration_n_saved

        if self.save_final is True:
            def _final_func(engine):
                return engine.state.iteration
            self.final_checkpoint = ModelCheckpoint(
                self.save_dir, file_prefix, score_function=_final_func, score_name='final_iteration',
                require_empty=False)
        if self.save_key_metric is True:
            def _score_func(engine):
                if isinstance(self.key_metric_name, str):
                    metric_name = self.key_metric_name
                elif hasattr(engine.state, "key_metric_name") and isinstance(engine.state.key_metric_name, str):
                    metric_name = engine.state.key_metric_name
                else:
                    raise ValueError("must provde key_metric_name to save best validation model.")
                return round(engine.state.metrics[metric_name], 4)
            self.key_metric_checkpoint = ModelCheckpoint(
                self.save_dir, file_prefix, score_function=_score_func, score_name='key_metric',
                n_saved=self.key_metric_n_saved, require_empty=False)
        if self.epoch_save_interval > 0:
            def _epoch_func(engine):
                return engine.state.epoch
            self.epoch_checkpoint = ModelCheckpoint(
                self.save_dir, file_prefix, score_function=_epoch_func, score_name='epoch',
                n_saved=self.epoch_n_saved, require_empty=False)
        if self.iteration_save_interval > 0:
            def _iteration_func(engine):
                return engine.state.iteration
            self.iteration_checkpoint = ModelCheckpoint(
                self.save_dir, file_prefix, score_function=_iteration_func, score_name='iteration',
                n_saved=self.iteration_n_saved, require_empty=False)

    def attach(self, engine):
        if self.logger is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self.completed, Events.COMPLETED):
            engine.add_event_handler(Events.COMPLETED, self.completed)
        if not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        if not engine.has_event_handler(self.exception_raised, Events.EXCEPTION_RAISED):
            engine.add_event_handler(Events.EXCEPTION_RAISED, self.exception_raised)

    def completed(self, engine):
        """Callback for train or validation/evaluation completed Event.
        Save final checkpoint if configure save_final is True.

        """
        if self.save_final is True:
            self.final_checkpoint(engine, self.save_dict)
            self.logger.info(f'Train completed, saved final checkpoint: {self.final_checkpoint.last_checkpoint}')

    def epoch_completed(self, engine):
        """Callback for train or validation/evaluation epoch completed Event.
        Save checkpoint if configure epoch_save_interval = N or save_key_metric is True.

        """
        if self.epoch_save_interval > 0 and engine.state.epoch % self.epoch_save_interval == 0:
            self.epoch_checkpoint(engine, self.save_dict)
            self.logger.info(f'Saved checkpoint at epoch: {engine.state.epoch}')
        if self.save_key_metric is True:
            self.key_metric_checkpoint(engine, self.save_dict)

    def iteration_completed(self, engine):
        """Callback for train or validation/evaluation iteration completed Event.
        Save checkpoint if configure iteration_save_interval = N

        """
        if self.iteration_save_interval > 0 and engine.state.iteration % self.iteration_save_interval == 0:
            self.iteration_checkpoint(engine, self.save_dict)
            self.logger.info(f'Saved checkpoint at iteration: {engine.state.iteration}')

    def exception_raised(self, engine, e):
        """Callback for train or validation/evaluation exception raised Event.
        Save current data as final checkpoint if configure save_final is True.

        """
        if self.save_final is True:
            self.final_checkpoint(engine, self.save_dict)
            self.logger.info(f'Exception_raised, saved exception checkpoint: {self.final_checkpoint.last_checkpoint}')
