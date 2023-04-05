import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union, TYPE_CHECKING

import torch

from monai.config import IgniteInfo
from monai.utils import optional_import, min_version, is_scalar

wandb, _ = optional_import("wandb")
Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")

if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.handlers import ModelCheckpoint
else:
    Engine, _ = optional_import(
        "ignite.engine",
        IgniteInfo.OPT_IMPORT_VERSION,
        min_version,
        "Engine",
        as_type="decorator",
    )
    ModelCheckpoint, _ = optional_import(
        "ignite.handlers", IgniteInfo.OPT_IMPORT_VERSION, min_version, "ModelCheckpoint"
    )

DEFAULT_TAG = "Loss"


class WandbStatsHandler:
    """
    `WandbStatsHandler` defines a set of Ignite Event-handlers for all the Weights & Biases logging
    logic. It can be used for any Ignite Engine(trainer, validator and evaluator) and support both
    epoch level and iteration level. The expected data source is Ignite `engine.state.output` and
    `engine.state.metrics`.
    Default behaviors:
        - When EPOCH_COMPLETED, write each dictionary item in `engine.state.metrics` to
            Weights & Biases.
        - When ITERATION_COMPLETED, write each dictionary item in
            `self.output_transform(engine.state.output)` to Weights & Biases.
    """

    def __init__(
        self,
        iteration_log: bool = True,
        epoch_log: bool = True,
        epoch_event_writer: Optional[Callable[[Engine, Any], Any]] = None,
        epoch_interval: int = 1,
        iteration_event_writer: Optional[Callable[[Engine, Any], Any]] = None,
        iteration_interval: int = 1,
        output_transform: Callable = lambda x: x[0],
        global_epoch_transform: Callable = lambda x: x,
        state_attributes: Optional[Sequence[str]] = None,
        tag_name: str = DEFAULT_TAG,
    ):
        """
        Args:
            iteration_log: whether to write data to Weights & Biases when iteration completed,
                default to `True`.
            epoch_log: whether to write data to Weights & Biases when epoch completed, default to
                `True`.
            epoch_event_writer: customized callable Weights & Biases writer for epoch level. Must
                accept parameter "engine" and "summary_writer", use default event writer if None.
            epoch_interval: the epoch interval at which the epoch_event_writer is called.
                Defaults to 1.
            iteration_event_writer: customized callable Weights & Biases writer for iteration level.
                Must accept parameter "engine" and "summary_writer", use default event writer if
                None.
            iteration_interval: the iteration interval at which the `iteration_event_writer` is
                called. Defaults to 1.
            output_transform: a callable that is used to transform the `ignite.engine.state.output`
                into a scalar to plot, or a dictionary of {key: scalar}. In the latter case, the
                output string will be formatted as key: value. By default this value plotting
                happens when every iteration completed. The default behavior is to print loss from
                output[0] as output is a decollated list and we replicated loss value for every item
                of the decollated list. `engine.state` and `output_transform` inherit from the ignite
                concept: https://pytorch.org/ignite/concepts.html#state, explanation and usage
                example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            global_epoch_transform: a callable that is used to customize global epoch number. For
                example, in evaluation, the evaluator engine might want to use trainer engines epoch
                number when plotting epoch vs metric curves.
            state_attributes: expected attributes from `engine.state`, if provided, will extract them
                when epoch completed.
            tag_name: when iteration output is a scalar, tag_name is used to plot, defaults to `'Loss'`.
        """
        if wandb.run is None:
            raise wandb.Error("You must call `wandb.init()` before WandbStatsHandler()")

        self.iteration_log = iteration_log
        self.epoch_log = epoch_log
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
        # Arguments:
            engine: `~ignite.engine.engine.Engine`.
                Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.iteration_log and not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=self.iteration_interval),
                self.iteration_completed,
            )
        if self.epoch_log and not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.epoch_interval), self.epoch_completed)

    def epoch_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation epoch completed Event. Write epoch level events
        to Weights & Biases, default values are from Ignite `engine.state.metrics` dict.
        # Arguments:
            engine: `~ignite.engine.engine.Engine`.
                Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.epoch_event_writer is not None:
            self.epoch_event_writer(engine)
        else:
            self._default_epoch_writer(engine)

    def iteration_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation iteration completed Event. Write iteration level
        events to Weighs & Biases, default values are from Ignite `engine.state.output`.
        # Arguments:
            engine: `~ignite.engine.engine.Engine`.
                Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self.iteration_event_writer is not None:
            self.iteration_event_writer(engine)
        else:
            self._default_iteration_writer(engine)

    def _default_epoch_writer(self, engine: Engine) -> None:
        """
        Execute epoch level event write operation. Default to write the values from Ignite
        `engine.state.metrics` dict and write the values of specified attributes of `engine.state`
        to [Weights & Biases](https://wandb.ai/site).
        # Arguments:
            engine: `~ignite.engine.engine.Engine`.
                Ignite Engine, it can be a trainer, validator or evaluator.
        """
        summary_dict = engine.state.metrics

        for key, value in summary_dict.items():
            if is_scalar(value):
                value = value.item() if isinstance(value, torch.Tensor) else value
                wandb.log({key: value})

        if self.state_attributes is not None:
            for attr in self.state_attributes:
                value = getattr(engine.state, attr, None)
                value = value.item() if isinstance(value, torch.Tensor) else value
                wandb.log({attr: value})

    def _default_iteration_writer(self, engine: Engine) -> None:
        """
        Execute iteration level event write operation based on Ignite `engine.state.output` data.
        Extract the values from `self.output_transform(engine.state.output)`. Since
        `engine.state.output` is a decollated list and we replicated the loss value for every item
        of the decollated list, the default behavior is to track the loss from `output[0]`.
        # Arguments:
            engine: `~ignite.engine.engine.Engine`.
                Ignite Engine, it can be a trainer, validator or evaluator.
        """
        loss = self.output_transform(engine.state.output)
        if loss is None:
            return  # do nothing if output is empty
        log_dict = dict()
        if isinstance(loss, dict):
            for key, value in loss.items():
                if not is_scalar(value):
                    warnings.warn(
                        "ignoring non-scalar output in WandbStatsHandler,"
                        " make sure `output_transform(engine.state.output)` returns"
                        " a scalar or dictionary of key and scalar pairs to avoid this warning."
                        " {}:{}".format(key, type(value))
                    )
                    continue  # not plot multi dimensional output
                log_dict[key] = value.item() if isinstance(value, torch.Tensor) else value
        elif is_scalar(loss):  # not printing multi dimensional output
            log_dict[self.tag_name] = loss.item() if isinstance(loss, torch.Tensor) else loss
        else:
            warnings.warn(
                "ignoring non-scalar output in WandbStatsHandler,"
                " make sure `output_transform(engine.state.output)` returns"
                " a scalar or a dictionary of key and scalar pairs to avoid this warning."
                " {}".format(type(loss))
            )

        wandb.log(log_dict)

    def close(self):
        """Close `WandbStatsHandler`"""
        wandb.finish()


class WandbModelCheckpointHandler(ModelCheckpoint):
    """`WandbModelCheckpointHandler` inherits from :class:`~ignite.handlers.ModelCheckpoint`, can be
    used to periodically save objects as
    [Weights & Biases artifacts](https://docs.wandb.ai/guides/artifacts).
    This handler expects two arguments:
        - a :class:`~ignite.engine.engine.Engine` object
        - a `dict` mapping names (`str`) to objects that should be saved to disk.
    """

    def __init__(
        self,
        dirname: Union[str, Path],
        filename_prefix: str = "",
        save_interval: Optional[int] = None,
        score_function: Optional[Callable] = None,
        score_name: Optional[str] = None,
        n_saved: Union[int, None] = 1,
        atomic: bool = True,
        require_empty: bool = True,
        create_dir: bool = True,
        save_as_state_dict: bool = True,
        global_step_transform: Optional[Callable] = None,
        archived: bool = False,
        filename_pattern: Optional[str] = None,
        include_self: bool = False,
        greater_or_equal: bool = False,
        save_on_rank: int = 0,
        **kwargs: Any,
    ):
        """
        Args:
            dirname: directory path where objects will be saved.
            filename_prefix: prefix for the file names to which objects will be saved. See Notes of
                :class:`~ignite.handlers.checkpoint.Checkpoint` for more details.
            save_interval: if not None, it should be a function taking a single argument, an
                :class:`~ignite.engine.engine.Engine` object, and return a score (`float`). Objects
                with highest scores will be retained.
            score_name: if `score_function` not None, it is possible to store its value using
                `score_name`. See Examples of :class:`~ignite.handlers.checkpoint.Checkpoint` for
                more details.
            n_saved: number of objects that should be kept on disk. Older files will be removed.
                If set to `None`, all objects are kept.
            atomic: if True, objects are serialized to a temporary file, and then moved to final
                destination, so that files are guaranteed to not be damaged (for example if
                exception occurs during saving).
            require_empty: if True, will raise exception if there are any files starting with
                `filename_prefix` in the directory `dirname`.
            create_dir: if True, will create directory `dirname` if it does not exist.
            global_step_transform: global step transform function to output a desired global step.
                Input of the function is `(engine, event_name)`. Output of function should be an
                integer. Default is None, `global_step` based on attached engine. If provided, uses
                function output as `global_step`. To setup global step from another engine, please
                use :meth:`~ignite.handlers.global_step_from_engine`.
            archived: deprecated argument as models saved by `torch.save` are already compressed.
            filename_pattern: if `filename_pattern` is provided, this pattern will be used to
                render checkpoint filenames. If the pattern is not defined, the default pattern
                would be used. See :class:`~ignite.handlers.checkpoint.Checkpoint` for details.
            include_self: whether to include the `state_dict` of this object in the checkpoint.
                If `True`, then there must not be another object in `to_save` with key
                `checkpointer`.
            greater_or_equal: if `True`, the latest equally scored model is stored. Otherwise, the
                first model. Default, `False`.
            save_on_rank: which rank to save the objects on, in the distributed configuration. Used
                to instantiate a :class:`~ignite.handlers.DiskSaver` and is also passed to the parent
                class.
            kwargs: accepted keyword arguments for `torch.save` or `xm.save` in `DiskSaver`.
        """
        if wandb.run is None:
            raise wandb.Error("You must call `wandb.init()` before WandbModelCheckpointHandler()")

        super().__init__(
            dirname,
            filename_prefix,
            save_interval,
            score_function,
            score_name,
            n_saved,
            atomic,
            require_empty,
            create_dir,
            save_as_state_dict,
            global_step_transform,
            archived,
            filename_pattern,
            include_self,
            greater_or_equal,
            save_on_rank,
            **kwargs,
        )

    def __call__(self, engine: Engine, to_save: Mapping):
        super().__call__(engine, to_save)
        artifact = wandb.Artifact(f"run-{wandb.run.id}-model", type="model")
        artifact.add_file(self.last_checkpoint)
        wandb.log_artifact(artifact)
