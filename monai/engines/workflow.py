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

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence, Sized
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from monai.engines.utils import IterationEvents, default_metric_cmp_fn, default_prepare_batch
from monai.transforms import Decollated
from monai.utils import IgniteInfo, ensure_tuple, is_scalar, min_version, optional_import

from .utils import engine_apply_transform

State, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "State")
Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )
    Metric, _ = optional_import(
        "ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric", as_type="decorator"
    )
    EventEnum, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum", as_type="decorator"
    )


class Workflow(Engine):
    """
    Workflow defines the core work process inheriting from Ignite engine.
    All trainer, validator and evaluator share this same workflow as base class,
    because they all can be treated as same Ignite engine loops.
    It initializes all the sharable data in Ignite engine.state.
    And attach additional processing logics to Ignite engine based on Event-Handler mechanism.

    Users should consider inheriting from `trainer` or `evaluator` to develop more trainers or evaluators.

    Args:
        device: an object representing the device on which to run.
        max_epochs: the total epoch number for engine to run, validator and evaluator have only 1 epoch.
        data_loader: Ignite engine use data_loader to run, must be Iterable or torch.DataLoader.
        epoch_length: number of iterations for one epoch, default to `len(data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, etc.
        amp: whether to enable auto-mixed-precision training or inference, default is False.
        event_names: additional custom ignite events that will register to the engine.
            new events can be a list of str or `ignite.engine.events.EventEnum`.
        event_to_attr: a dictionary to map an event to a state attribute, then add to `engine.state`.
            for more details, check: https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html
            #ignite.engine.engine.Engine.register_events.
        decollate: whether to decollate the batch-first data to a list of data after model computation,
            recommend `decollate=True` when `postprocessing` uses components from `monai.transforms`.
            default to `True`.
        to_kwargs: dict of other args for `prepare_batch` API when converting the input data, except for
            `device`, `non_blocking`.
        amp_kwargs: dict of the args for `torch.cuda.amp.autocast()` API, for more details:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.autocast.

    Raises:
        TypeError: When ``data_loader`` is not a ``torch.utils.data.DataLoader``.
        TypeError: When ``key_metric`` is not a ``Optional[dict]``.
        TypeError: When ``additional_metrics`` is not a ``Optional[dict]``.

    """

    def __init__(
        self,
        device: torch.device | str,
        max_epochs: int,
        data_loader: Iterable | DataLoader,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        postprocessing: Callable | None = None,
        key_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        handlers: Sequence | None = None,
        amp: bool = False,
        event_names: list[str | EventEnum | type[EventEnum]] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(self._iteration if iteration_update is None else iteration_update)

        if isinstance(data_loader, DataLoader):
            sampler = getattr(data_loader, "sampler", None)

            # set the epoch value for DistributedSampler objects when an epoch starts
            if isinstance(sampler, DistributedSampler):

                @self.on(Events.EPOCH_STARTED)
                def set_sampler_epoch(engine: Engine) -> None:
                    sampler.set_epoch(engine.state.epoch)

        # if the epoch_length isn't given, attempt to get it from the length of the data loader
        if epoch_length is None and isinstance(data_loader, Sized):
            try:
                epoch_length = len(data_loader)
            except TypeError:  # raised when data_loader has an iterable dataset with no length, or is some other type
                pass  # deliberately leave epoch_length as None

        # set all sharable data for the workflow based on Ignite engine.state
        self.state: Any = State(
            rank=dist.get_rank() if dist.is_available() and dist.is_initialized() else 0,
            seed=0,
            iteration=0,
            epoch=0,
            max_epochs=max_epochs,
            epoch_length=epoch_length,  # None when the dataset is iterable and so has no length
            output=None,
            batch=None,
            metrics={},
            metric_details={},
            dataloader=None,
            device=device if isinstance(device, torch.device) or device is None else torch.device(device),
            key_metric_name=None,  # we can set many metrics, only use key_metric to compare and save the best model
            best_metric=-1,
            best_metric_epoch=-1,
        )
        self.data_loader = data_loader
        self.non_blocking = non_blocking
        self.prepare_batch = prepare_batch
        self.metric_cmp_fn = metric_cmp_fn
        self.amp = amp
        self.to_kwargs = {} if to_kwargs is None else to_kwargs
        self.amp_kwargs = {} if amp_kwargs is None else amp_kwargs
        self.scaler: torch.cuda.amp.GradScaler | None = None

        if event_names is None:
            event_names = [IterationEvents]
        else:
            if not isinstance(event_names, list):
                raise ValueError("`event_names` must be a list of strings or EventEnums.")
            event_names += [IterationEvents]
        for name in event_names:
            if isinstance(name, (str, EventEnum)):
                self.register_events(name, event_to_attr=event_to_attr)  # type: ignore[arg-type]
            elif issubclass(name, EventEnum):
                self.register_events(*name, event_to_attr=event_to_attr)
            else:
                raise ValueError("`event_names` must be a list of strings or EventEnums.")

        if decollate:
            self._register_decollate()

        if postprocessing is not None:
            # tips: if `decollate=False` and `postprocessing` is MONAI transforms, it may not work well
            # because all the MONAI transforms expect `channel-first` data
            self._register_postprocessing(postprocessing)
        if key_metric is not None:
            self._register_metrics(key_metric, additional_metrics)
        if handlers is not None:
            self._register_handlers(handlers)

    def _register_decollate(self):
        """
        Register the decollate operation for batch data, will execute after model forward and loss forward.

        """

        @self.on(IterationEvents.MODEL_COMPLETED)
        def _decollate_data(engine: Engine) -> None:
            # replicate the scalar values to make sure all the items have batch dimension, then decollate
            transform = Decollated(keys=None, detach=True)
            if isinstance(engine.state.batch, (list, dict)):
                engine.state.batch = transform(engine.state.batch)
            if isinstance(engine.state.output, (list, dict)):
                engine.state.output = transform(engine.state.output)

    def _register_postprocessing(self, posttrans: Callable) -> None:
        """
        Register the postprocessing logic to the engine, will execute them as a chain when iteration completed.

        """

        @self.on(IterationEvents.MODEL_COMPLETED)
        def _run_postprocessing(engine: Engine) -> None:
            if not isinstance(engine.state.batch, list) or not isinstance(engine.state.output, list):
                engine.state.batch, engine.state.output = engine_apply_transform(
                    batch=engine.state.batch, output=engine.state.output, transform=posttrans
                )
            else:
                for i, (b, o) in enumerate(zip(engine.state.batch, engine.state.output)):
                    engine.state.batch[i], engine.state.output[i] = engine_apply_transform(b, o, posttrans)

    def _register_metrics(self, k_metric: dict, add_metrics: dict | None = None) -> None:
        """
        Register the key metric and additional metrics to the engine, supports ignite Metrics.

        """
        if not isinstance(k_metric, dict):
            raise TypeError(f"`key_metric` must be None or a dict but is {type(k_metric).__name__}.")
        self.state.key_metric_name = list(k_metric.keys())[0]
        metrics = dict(k_metric)
        if add_metrics is not None and len(add_metrics) > 0:
            if not isinstance(add_metrics, dict):
                raise TypeError(f"Additional metrics must be None or a dict but is {type(add_metrics).__name__}.")
            metrics.update(add_metrics)
        for name, metric in metrics.items():
            metric.attach(self, name)

        @self.on(Events.EPOCH_COMPLETED)
        def _compare_metrics(engine: Workflow) -> None:
            key_metric_name = engine.state.key_metric_name
            if key_metric_name is not None:
                current_val_metric = engine.state.metrics[key_metric_name]
                if not is_scalar(current_val_metric):
                    warnings.warn(
                        "Key metric is not a scalar value, skip the metric comparison with the current best metric."
                        "Please set other metrics as the key metric, or change the `reduction` mode to 'mean'."
                    )
                    return

                if engine.state.best_metric_epoch == -1 or self.metric_cmp_fn(
                    current_val_metric, engine.state.best_metric
                ):
                    self.logger.info(f"Got new best metric of {key_metric_name}: {current_val_metric}")
                    engine.state.best_metric = current_val_metric
                    engine.state.best_metric_epoch = engine.state.epoch

    def _register_handlers(self, handlers: Sequence) -> None:
        """
        Register the handlers to the engine, supports ignite Handlers with `attach` API.

        """
        handlers_ = ensure_tuple(handlers)
        for handler in handlers_:
            handler.attach(self)

    def run(self) -> None:  # type: ignore[override]
        """
        Execute training, validation or evaluation based on Ignite Engine.
        """
        if self.state.epoch_length == 0:
            warnings.warn(
                "`dataloader` is empty or the specified `epoch_length` is 0, skip the `run`."
                " If running distributed training, the program may hang in `all-gather`, `all-reduce`, etc."
                " because not all the ranks run the same computation logic."
            )
            return
        super().run(data=self.data_loader, max_epochs=self.state.max_epochs)

    def _iteration(self, engine: Any, batchdata: dict[str, torch.Tensor]) -> dict:
        """
        Abstract callback function for the processing logic of 1 iteration in Ignite Engine.
        Need subclass to implement different logics, like SupervisedTrainer/Evaluator, GANTrainer, etc.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def get_stats(self, *vars):
        """
        Get the statistics information of the workflow process.

        Args:
            vars: variables name in the `self.state`, will use the variable name as the key
                and the state content as the value. if the variable doesn't exist, default value is `None`.

        """
        return {k: getattr(self.state, k, None) for k in vars}
