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

import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np
import torch

from monai.config import IgniteInfo
from monai.utils import is_scalar, min_version, optional_import
from monai.visualize import plot_2d_or_3d_image

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")

if TYPE_CHECKING:
    from ignite.engine import Engine
    from torch.utils.tensorboard import SummaryWriter
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    SummaryWriter, _ = optional_import("torch.utils.tensorboard", name="SummaryWriter")

DEFAULT_TAG = "Loss"


class TensorBoardHandler:
    """
    Base class for the handlers to write data into TensorBoard.

    Args:
        summary_writer: user can specify TensorBoard or TensorBoardX SummaryWriter,
            default to create a new TensorBoard writer.
        log_dir: if using default SummaryWriter, write logs to this directory, default is `./runs`.

    """

    def __init__(self, summary_writer: Optional[SummaryWriter] = None, log_dir: str = "./runs"):
        if summary_writer is None:
            self._writer = SummaryWriter(log_dir=log_dir)
            self.internal_writer = True
        else:
            self._writer = summary_writer
            self.internal_writer = False

    def attach(self, engine: Engine) -> None:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def close(self):
        """
        Close the summary writer if created in this TensorBoard handler.

        """
        if self.internal_writer:
            self._writer.close()


class TensorBoardStatsHandler(TensorBoardHandler):
    """
    TensorBoardStatsHandler defines a set of Ignite Event-handlers for all the TensorBoard logics.
    It can be used for any Ignite Engine(trainer, validator and evaluator).
    And it can support both epoch level and iteration level with pre-defined TensorBoard event writer.
    The expected data source is Ignite ``engine.state.output`` and ``engine.state.metrics``.

    Default behaviors:
        - When EPOCH_COMPLETED, write each dictionary item in
          ``engine.state.metrics`` to TensorBoard.
        - When ITERATION_COMPLETED, write each dictionary item in
          ``self.output_transform(engine.state.output)`` to TensorBoard.

    Usage example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unet_segmentation_3d_ignite.ipynb.

    """

    def __init__(
        self,
        summary_writer: Optional[SummaryWriter] = None,
        log_dir: str = "./runs",
        epoch_event_writer: Optional[Callable[[Engine, SummaryWriter], Any]] = None,
        epoch_interval: int = 1,
        iteration_event_writer: Optional[Callable[[Engine, SummaryWriter], Any]] = None,
        iteration_interval: int = 1,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Optional[Sequence[str]] = None,
        tag_name: str = DEFAULT_TAG,
    ) -> None:
        """
        Args:
            summary_writer: user can specify TensorBoard or TensorBoardX SummaryWriter,
                default to create a new TensorBoard writer.
            log_dir: if using default SummaryWriter, write logs to this directory, default is `./runs`.
            epoch_event_writer: customized callable TensorBoard writer for epoch level.
                Must accept parameter "engine" and "summary_writer", use default event writer if None.
            epoch_interval: the epoch interval at which the epoch_event_writer is called. Defaults to 1.
            iteration_event_writer: customized callable TensorBoard writer for iteration level.
                Must accept parameter "engine" and "summary_writer", use default event writer if None.
            iteration_interval: the iteration interval at which the iteration_event_writer is called. Defaults to 1.
            output_transform: a callable that is used to transform the
                ``ignite.engine.state.output`` into a scalar to plot, or a dictionary of {key: scalar}.
                In the latter case, the output string will be formatted as key: value.
                By default this value plotting happens when every iteration completed.
                The default behavior is to print loss from output[0] as output is a decollated list
                and we replicated loss value for every item of the decollated list.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            global_epoch_transform: a callable that is used to customize global epoch number.
                For example, in evaluation, the evaluator engine might want to use trainer engines epoch number
                when plotting epoch vs metric curves.
            state_attributes: expected attributes from `engine.state`, if provided, will extract them
                when epoch completed.
            tag_name: when iteration output is a scalar, tag_name is used to plot, defaults to ``'Loss'``.
        """
        super().__init__(summary_writer=summary_writer, log_dir=log_dir)
        self.epoch_event_writer = epoch_event_writer
        self.epoch_interval = epoch_interval
        self.iteration_event_writer = iteration_event_writer
        self.iteration_interval = iteration_interval
        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.state_attributes = state_attributes
        self.tag_name = tag_name

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=self.iteration_interval), self.iteration_completed
            )
        if not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.epoch_interval), self.epoch_completed)

    def epoch_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation epoch completed Event.
        Write epoch level events, default values are from Ignite `engine.state.metrics` dict.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.epoch_event_writer is not None:
            self.epoch_event_writer(engine, self._writer)
        else:
            self._default_epoch_writer(engine, self._writer)

    def iteration_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation iteration completed Event.
        Write iteration level events, default values are from Ignite `engine.state.output`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if self.iteration_event_writer is not None:
            self.iteration_event_writer(engine, self._writer)
        else:
            self._default_iteration_writer(engine, self._writer)

    def _write_scalar(self, _engine: Engine, writer: SummaryWriter, tag: str, value: Any, step: int) -> None:
        """
        Write scale value into TensorBoard.
        Default to call `SummaryWriter.add_scalar()`.

        Args:
            _engine: Ignite Engine, unused argument.
            writer: TensorBoard or TensorBoardX writer, passed or created in TensorBoardHandler.
            tag: tag name in the TensorBoard.
            value: value of the scalar data for current step.
            step: index of current step.

        """
        writer.add_scalar(tag, value, step)

    def _default_epoch_writer(self, engine: Engine, writer: SummaryWriter) -> None:
        """
        Execute epoch level event write operation.
        Default to write the values from Ignite `engine.state.metrics` dict and
        write the values of specified attributes of `engine.state`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            writer: TensorBoard or TensorBoardX writer, passed or created in TensorBoardHandler.

        """
        current_epoch = self.global_epoch_transform(engine.state.epoch)
        summary_dict = engine.state.metrics
        for name, value in summary_dict.items():
            if is_scalar(value):
                self._write_scalar(engine, writer, name, value, current_epoch)

        if self.state_attributes is not None:
            for attr in self.state_attributes:
                self._write_scalar(engine, writer, attr, getattr(engine.state, attr, None), current_epoch)
        writer.flush()

    def _default_iteration_writer(self, engine: Engine, writer: SummaryWriter) -> None:
        """
        Execute iteration level event write operation based on Ignite `engine.state.output` data.
        Extract the values from `self.output_transform(engine.state.output)`.
        Since `engine.state.output` is a decollated list and we replicated the loss value for every item
        of the decollated list, the default behavior is to track the loss from `output[0]`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            writer: TensorBoard  or TensorBoardX writer, passed or created in TensorBoardHandler.

        """
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # do nothing if output is empty
        if isinstance(loss, dict):
            for name in sorted(loss):
                value = loss[name]
                if not is_scalar(value):
                    warnings.warn(
                        "ignoring non-scalar output in TensorBoardStatsHandler,"
                        " make sure `output_transform(engine.state.output)` returns"
                        " a scalar or dictionary of key and scalar pairs to avoid this warning."
                        " {}:{}".format(name, type(value))
                    )
                    continue  # not plot multi dimensional output
                self._write_scalar(
                    _engine=engine,
                    writer=writer,
                    tag=name,
                    value=value.item() if isinstance(value, torch.Tensor) else value,
                    step=engine.state.iteration,
                )
        elif is_scalar(loss):  # not printing multi dimensional output
            self._write_scalar(
                _engine=engine,
                writer=writer,
                tag=self.tag_name,
                value=loss.item() if isinstance(loss, torch.Tensor) else loss,
                step=engine.state.iteration,
            )
        else:
            warnings.warn(
                "ignoring non-scalar output in TensorBoardStatsHandler,"
                " make sure `output_transform(engine.state.output)` returns"
                " a scalar or a dictionary of key and scalar pairs to avoid this warning."
                " {}".format(type(loss))
            )
        writer.flush()


class TensorBoardImageHandler(TensorBoardHandler):
    """
    TensorBoardImageHandler is an Ignite Event handler that can visualize images, labels and outputs as 2D/3D images.
    2D output (shape in Batch, channel, H, W) will be shown as simple image using the first element in the batch,
    for 3D to ND output (shape in Batch, channel, H, W, D) input, each of ``self.max_channels`` number of images'
    last three dimensions will be shown as animated GIF along the last axis (typically Depth).
    And if writer is from TensorBoardX, data has 3 channels and `max_channels=3`, will plot as RGB video.

    It can be used for any Ignite Engine (trainer, validator and evaluator).
    User can easily add it to engine for any expected Event, for example: ``EPOCH_COMPLETED``,
    ``ITERATION_COMPLETED``. The expected data source is ignite's ``engine.state.batch`` and ``engine.state.output``.

    Default behavior:
        - Show y_pred as images (GIF for 3D) on TensorBoard when Event triggered,
        - Need to use ``batch_transform`` and ``output_transform`` to specify
          how many images to show and show which channel.
        - Expects ``batch_transform(engine.state.batch)`` to return data
          format: (image[N, channel, ...], label[N, channel, ...]).
        - Expects ``output_transform(engine.state.output)`` to return a torch
          tensor in format (y_pred[N, channel, ...], loss).

    Usage example is available in the tutorial:
    https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/unet_segmentation_3d_ignite.ipynb.

    """

    def __init__(
        self,
        summary_writer: Optional[SummaryWriter] = None,
        log_dir: str = "./runs",
        interval: int = 1,
        epoch_level: bool = True,
        batch_transform: Callable = lambda x: x,
        output_transform: Callable = lambda x: x,
        global_iter_transform: Callable = lambda x: x,
        index: int = 0,
        max_channels: int = 1,
        frame_dim: int = -3,
        max_frames: int = 64,
    ) -> None:
        """
        Args:
            summary_writer: user can specify TensorBoard or TensorBoardX SummaryWriter,
                default to create a new TensorBoard writer.
            log_dir: if using default SummaryWriter, write logs to this directory, default is `./runs`.
            interval: plot content from engine.state every N epochs or every N iterations, default is 1.
            epoch_level: plot content from engine.state every N epochs or N iterations. `True` is epoch level,
                `False` is iteration level.
            batch_transform: a callable that is used to extract `image` and `label` from `ignite.engine.state.batch`,
                then construct `(image, label)` pair. for example: if `ignite.engine.state.batch` is `{"image": xxx,
                "label": xxx, "other": xxx}`, `batch_transform` can be `lambda x: (x["image"], x["label"])`.
                will use the result to plot image from `result[0][index]` and plot label from `result[1][index]`.
                `engine.state` and `batch_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            output_transform: a callable that is used to extract the `predictions` data from
                `ignite.engine.state.output`, will use the result to plot output from `result[index]`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            global_iter_transform: a callable that is used to customize global step number for TensorBoard.
                For example, in evaluation, the evaluator engine needs to know current epoch from trainer.
            index: plot which element in a data batch, default is the first element.
            max_channels: number of channels to plot.
            frame_dim: if plotting 3D image as GIF, specify the dimension used as frames,
                expect input data shape as `NCHWD`, default to `-3` (the first spatial dim)
            max_frames: if plot 3D RGB image as video in TensorBoardX, set the FPS to `max_frames`.
        """
        super().__init__(summary_writer=summary_writer, log_dir=log_dir)
        self.interval = interval
        self.epoch_level = epoch_level
        self.batch_transform = batch_transform
        self.output_transform = output_transform
        self.global_iter_transform = global_iter_transform
        self.index = index
        self.frame_dim = frame_dim
        self.max_frames = max_frames
        self.max_channels = max_channels

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.interval), self)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.interval), self)

    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        Raises:
            TypeError: When ``output_transform(engine.state.output)[0]`` type is not in
                ``Optional[Union[numpy.ndarray, torch.Tensor]]``.
            TypeError: When ``batch_transform(engine.state.batch)[1]`` type is not in
                ``Optional[Union[numpy.ndarray, torch.Tensor]]``.
            TypeError: When ``output_transform(engine.state.output)`` type is not in
                ``Optional[Union[numpy.ndarray, torch.Tensor]]``.

        """
        step = self.global_iter_transform(engine.state.epoch if self.epoch_level else engine.state.iteration)
        show_images = self.batch_transform(engine.state.batch)[0][self.index]
        if isinstance(show_images, torch.Tensor):
            show_images = show_images.detach().cpu().numpy()
        if show_images is not None:
            if not isinstance(show_images, np.ndarray):
                raise TypeError(
                    "output_transform(engine.state.output)[0] must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_images).__name__}."
                )
            plot_2d_or_3d_image(
                # add batch dim and plot the first item
                data=show_images[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                frame_dim=self.frame_dim,
                max_frames=self.max_frames,
                tag="input_0",
            )

        show_labels = self.batch_transform(engine.state.batch)[1][self.index]
        if isinstance(show_labels, torch.Tensor):
            show_labels = show_labels.detach().cpu().numpy()
        if show_labels is not None:
            if not isinstance(show_labels, np.ndarray):
                raise TypeError(
                    "batch_transform(engine.state.batch)[1] must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_labels).__name__}."
                )
            plot_2d_or_3d_image(
                data=show_labels[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                frame_dim=self.frame_dim,
                max_frames=self.max_frames,
                tag="input_1",
            )

        show_outputs = self.output_transform(engine.state.output)[self.index]
        if isinstance(show_outputs, torch.Tensor):
            show_outputs = show_outputs.detach().cpu().numpy()
        if show_outputs is not None:
            if not isinstance(show_outputs, np.ndarray):
                raise TypeError(
                    "output_transform(engine.state.output) must be None or one of "
                    f"(numpy.ndarray, torch.Tensor) but is {type(show_outputs).__name__}."
                )
            plot_2d_or_3d_image(
                data=show_outputs[None],
                step=step,
                writer=self._writer,
                index=0,
                max_channels=self.max_channels,
                frame_dim=self.frame_dim,
                max_frames=self.max_frames,
                tag="output",
            )

        self._writer.flush()
