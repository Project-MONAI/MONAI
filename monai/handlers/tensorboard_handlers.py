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

import torch
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Engine, Events
from monai.visualize import img2tensorboard


class TensorBoardStatsHandler(object):
    """TensorBoardStatsHandler defines a set of Ignite Event-handlers for all the TensorBoard logics.
    It's can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support both epoch level and iteration level with pre-defined TensorBoard event writer.
    The expected data source is ignite `engine.state.output` and `engine.state.metrics`.
    Default behaviors:
    (1) Write metrics to TensorBoard when EPOCH_COMPLETED.
    (2) Write loss to TensorBoard when ITERATION_COMPLETED, expected output format is (y_pred, loss).

    """
    def __init__(self,
                 summary_writer=None,
                 epoch_event_writer=None,
                 iteration_event_writer=None,
                 batch_transform=lambda x: x,
                 output_transform=lambda x: x,
                 global_epoch_transform=None):
        """
        Args:
            summary_writer (SummaryWriter): user can specify TensorBoard SummaryWriter,
                default to create a new writer.
            epoch_event_writer (Callable): customized callable TensorBoard writer for epoch level.
                must accept parameter "engine" and "summary_writer", use default event writer if None.
            iteration_event_writer (Callable): custimized callable TensorBoard writer for iteration level.
                must accept parameter "engine" and "summary_writer", use default event writer if None.
            batch_transform (Callable): a callable that is used to transform the
                ignite.engine.batch into expected format to extract several label data.
            output_transform (Callable): a callable that is used to transform the
                ignite.engine.output into expected format to extract several output data.
            global_epoch_transform (Callable): a callable that is used to customize global epoch number.
                For example, in evaluation, the evaluator engine needs to know current epoch from trainer.

        """
        self._writer = SummaryWriter() if summary_writer is None else summary_writer
        self.epoch_event_writer = epoch_event_writer
        self.iteration_event_writer = iteration_event_writer
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform


    def attach(self, engine: Engine):
        """Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        if not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)

    def epoch_completed(self, engine: Engine):
        """handler for train or validation/evaluation epoch completed Event.
        Write epoch level events, default values are from ignite state.metrics dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.epoch_event_writer is not None:
            self.epoch_event_writer(engine, self._writer)
        else:
            self._default_epoch_writer(engine, self._writer)

    def iteration_completed(self, engine: Engine):
        """handler for train or validation/evaluation iteration completed Event.
        Write iteration level events, default values are from ignite state.logs dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.iteration_event_writer is not None:
            self.iteration_event_writer(engine, self._writer)
        else:
            self._default_iteration_writer(engine, self._writer)

    def _default_epoch_writer(self, engine: Engine, writer: SummaryWriter):
        """Execute epoch level event write operation based on Ignite engine.state data.
        Default is to write the values from ignite state.metrics dict.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.
            writer (SummaryWriter): TensorBoard writer, created in TensorBoardHandler.

        """
        current_epoch = self.global_epoch_transform() if self.global_epoch_transform is not None \
            else engine.state.epoch
        summary_dict = engine.state.metrics
        for name, value in summary_dict.items():
            writer.add_scalar(name, value, current_epoch)
        writer.flush()

    def _default_iteration_writer(self, engine: Engine, writer: SummaryWriter):
        """Execute iteration level event write operation based on Ignite engine.state data.
        Default is to write the loss value of current iteration.

        Args:
            engine (ignite.engine): Ignite Engine, it can be a trainer, validator or evaluator.
            writer (SummaryWriter): TensorBoard writer, created in TensorBoardHandler.

        """
        loss = self.output_transform(engine.state.output)[1]
        if loss is None or (torch.is_tensor(loss) and len(loss.shape) > 0):
            return  # not record multi dimensional output
            writer.add_scalar('Loss', loss.item() if torch.is_tensor(loss) else loss, engine.state.iteration)
            writer.flush()


class TensorBoardImageHandler(object):
    """TensorBoardImageHandler is an ignite Event handler that can draw images, labels and outputs as 2D/3D images.
    2D output will be shown as simple image, 3D output will be shown as GIF image along the last axis(typically Depth)
    It's can be used for any Ignite Engine(trainer, validator and evaluator).
    User can easily added it to engine for any expected Event, for example: EPOCH_COMPLETED, ITERATION_COMPLETED.
    The expected data source is ignite `engine.state.batch` and `engine.state.output`.
    Default behavior:
    Show y_pred as images(GIF for 3D) on TensorBoard when Event triggered,
    need to use `batch_transform` and `output_transform` to specify how many images to show and show which channel.
    Expected batch data format is (image[N, channel, ...], label[N, channel, ...]),
    Expected output data format is (y_pred[N, channel, ...], loss).
    """

    def __init__(self,
                 summary_writer=None,
                 batch_transform=lambda x: x,
                 output_transform=lambda x: x,
                 global_step_transform=None):
        """
        Args:
            summary_writer (SummaryWriter): user can specify TensorBoard SummaryWriter,
                default to create a new writer.
            batch_transform (Callable): a callable that is used to transform the
                ignite.engine.batch into expected format to extract several label data.
            output_transform (Callable): a callable that is used to transform the
                ignite.engine.output into expected format to extract several output data.
            global_step_transform (Callable): a callable that is used to customize global step number for TensorBoard.
                For example, in evaluation, the evaluator engine needs to know current epoch from trainer.

        """
        self._writer = SummaryWriter() if summary_writer is None else summary_writer
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_step_transform = global_step_transform

    def __call__(self, engine):
        step = self.global_step_transform() if self.global_step_transform is not None else None
        show_images = self.batch_transform(engine.state.batch)[0]
        show_labels = self.batch_transform(engine.state.batch)[1]
        show_outputs = self.output_transform(engine.state.output)[0].detach().cpu()
        if show_images is not None and show_labels is not None and show_outputs is not None:
            ones = torch.ones(show_labels[0].shape, dtype=torch.int32)
            if len(show_labels.shape) == 5:
                assert show_images.shape[1] == show_labels.shape[1] == show_outputs.shape[1] == 1, \
                    '3D images must select only 1 channel to show.'
                for i in range(len(show_labels)):
                    img2tensorboard.add_animated_gif(self._writer, 'image' + str(i), show_images[i], 64, 255, step)
                    label_tensor = torch.where(show_labels[i] > 0, ones, show_labels[i])
                    img2tensorboard.add_animated_gif(self._writer, 'label' + str(i), label_tensor, 64, 255, step)
                    img2tensorboard.add_animated_gif(self._writer, 'output' + str(i), show_outputs[i], 64, 255, step)
            elif len(show_labels.shape) == 4:
                for i in range(len(show_labels)):
                    self._writer.add_image('image' + str(i), show_images[i], step)
                    label_tensor = torch.where(show_labels[i] > 0, ones, show_labels[i])
                    self._writer.add_image('label' + str(i), label_tensor, step)
                    self._writer.add_image('output' + str(i), show_outputs[i], step)
            else:
                raise ValueError('unsupported input data shape.')
            self._writer.flush()
