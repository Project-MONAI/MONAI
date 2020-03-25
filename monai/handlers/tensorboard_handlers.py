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

import numpy as np
import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Engine, Events
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.utils.misc import is_scalar

DEFAULT_TAG = 'Loss'


class TensorBoardStatsHandler(object):
    """TensorBoardStatsHandler defines a set of Ignite Event-handlers for all the TensorBoard logics.
    It's can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support both epoch level and iteration level with pre-defined TensorBoard event writer.
    The expected data source is ignite ``engine.state.output`` and ``engine.state.metrics``.

    Default behaviors:
        - When EPOCH_COMPLETED, write each dictionary item in
          ``engine.state.metrics`` to TensorBoard.
        - When ITERATION_COMPELTED, write each dictionary item in
          ``self.output_transform(engine.state.output)`` to TensorBoard.
    """

    def __init__(self,
                 summary_writer=None,
                 epoch_event_writer=None,
                 iteration_event_writer=None,
                 output_transform=lambda x: x,
                 global_epoch_transform=lambda x: x,
                 tag_name=DEFAULT_TAG):
        """
        Args:
            summary_writer (SummaryWriter): user can specify TensorBoard SummaryWriter,
                default to create a new writer.
            epoch_event_writer (Callable): customized callable TensorBoard writer for epoch level.
                must accept parameter "engine" and "summary_writer", use default event writer if None.
            iteration_event_writer (Callable): custimized callable TensorBoard writer for iteration level.
                must accept parameter "engine" and "summary_writer", use default event writer if None.
            output_transform (Callable): a callable that is used to transform the
                ``ignite.engine.output`` into a scalar to plot, or a dictionary of {key: scalar}.
                in the latter case, the output string will be formated as key: value.
                by default this value plotting happens when every iteration completed.
            global_epoch_transform (Callable): a callable that is used to customize global epoch number.
                For example, in evaluation, the evaluator engine might want to use trainer engines epoch number
                when plotting epoch vs metric curves.
            tag_name (string): when iteration output is a scalar, tag_name is used to plot, defaults to ``'Loss'``.
        """
        self._writer = SummaryWriter() if summary_writer is None else summary_writer
        self.epoch_event_writer = epoch_event_writer
        self.iteration_event_writer = iteration_event_writer
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.tag_name = tag_name

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
        current_epoch = self.global_epoch_transform(engine.state.epoch)
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
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # do nothing if output is empty
        if isinstance(loss, dict):
            for name in sorted(loss):
                value = loss[name]
                if not is_scalar(value):
                    warnings.warn('ignoring non-scalar output in TensorBoardStatsHandler,'
                                  ' make sure `output_transform(engine.state.output)` returns'
                                  ' a scalar or dictionary of key and scalar pairs to avoid this warning.'
                                  ' {}:{}'.format(name, type(value)))
                    continue  # not plot multi dimensional output
                writer.add_scalar(name, value.item() if torch.is_tensor(value) else value, engine.state.iteration)
        elif is_scalar(loss):  # not printing multi dimensional output
            writer.add_scalar(self.tag_name, loss.item() if torch.is_tensor(loss) else loss, engine.state.iteration)
        else:
            warnings.warn('ignoring non-scalar output in TensorBoardStatsHandler,'
                          ' make sure `output_transform(engine.state.output)` returns'
                          ' a scalar or a dictionary of key and scalar pairs to avoid this warning.'
                          ' {}'.format(type(loss)))
        writer.flush()


class TensorBoardImageHandler(object):
    """TensorBoardImageHandler is an ignite Event handler that can visualise images, labels and outputs as 2D/3D images.
    2D output (shape in Batch, channel, H, W) will be shown as simple image using the first element in the batch,
    for 3D to ND output (shape in Batch, channel, H, W, D) input, each of ``self.max_channels`` number of images'
    last three dimensions will be shown as animated GIF along the last axis (typically Depth).

    It's can be used for any Ignite Engine (trainer, validator and evaluator).
    User can easily added it to engine for any expected Event, for example: ``EPOCH_COMPLETED``,
    ``ITERATION_COMPLETED``. The expected data source is ignite's ``engine.state.batch`` and ``engine.state.output``.

    Default behavior:
        - Show y_pred as images (GIF for 3D) on TensorBoard when Event triggered,
        - need to use ``batch_transform`` and ``output_transform`` to specify
          how many images to show and show which channel.
        - Expects ``batch_transform(engine.state.batch)`` to return data
          format: (image[N, channel, ...], label[N, channel, ...]).
        - Expects ``output_transform(engine.state.output)`` to return a torch
          tensor in format (y_pred[N, channel, ...], loss).

     """

    def __init__(self,
                 summary_writer=None,
                 batch_transform=lambda x: x,
                 output_transform=lambda x: x,
                 global_iter_transform=lambda x: x,
                 index=0,
                 max_channels=1,
                 max_frames=64):
        """
        Args:
            summary_writer (SummaryWriter): user can specify TensorBoard SummaryWriter,
                default to create a new writer.
            batch_transform (Callable): a callable that is used to transform the
                ``ignite.engine.batch`` into expected format to extract several label data.
            output_transform (Callable): a callable that is used to transform the
                ``ignite.engine.output`` into expected format to extract several output data.
            global_iter_transform (Callable): a callable that is used to customize global step number for TensorBoard.
                For example, in evaluation, the evaluator engine needs to know current epoch from trainer.
            index (int): plot which element in a data batch, default is the first element.
            max_channels (int): number of channels to plot.
            max_frames (int): number of frames for 2D-t plot.
        """
        self._writer = SummaryWriter() if summary_writer is None else summary_writer
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_iter_transform = global_iter_transform
        self.index = index
        self.max_frames = max_frames
        self.max_channels = max_channels

    def __call__(self, engine):
        step = self.global_iter_transform(engine.state.iteration)

        show_images = self.batch_transform(engine.state.batch)[0]
        if torch.is_tensor(show_images):
            show_images = show_images.detach().cpu().numpy()
        if show_images is not None:
            if not isinstance(show_images, np.ndarray):
                raise ValueError('output_transform(engine.state.output)[0] must be an ndarray or tensor.')
            plot_2d_or_3d_image(show_images, step, self._writer, self.index,
                                self.max_channels, self.max_frames, 'input_0')

        show_labels = self.batch_transform(engine.state.batch)[1]
        if torch.is_tensor(show_labels):
            show_labels = show_labels.detach().cpu().numpy()
        if show_labels is not None:
            if not isinstance(show_labels, np.ndarray):
                raise ValueError('batch_transform(engine.state.batch)[1] must be an ndarray or tensor.')
            plot_2d_or_3d_image(show_labels, step, self._writer, self.index,
                                self.max_channels, self.max_frames, 'input_1')

        show_outputs = self.output_transform(engine.state.output)
        if torch.is_tensor(show_outputs):
            show_outputs = show_outputs.detach().cpu().numpy()
        if show_outputs is not None:
            if not isinstance(show_outputs, np.ndarray):
                raise ValueError('output_transform(engine.state.output) must be an ndarray or tensor.')
            plot_2d_or_3d_image(show_outputs, step, self._writer, self.index,
                                self.max_channels, self.max_frames, 'output')

        self._writer.flush()
