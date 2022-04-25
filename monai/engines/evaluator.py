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

from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from monai.config import IgniteInfo, KeysCollection
from monai.engines.utils import IterationEvents, default_metric_cmp_fn, default_prepare_batch
from monai.engines.workflow import Workflow
from monai.inferers import Inferer, SimpleInferer
from monai.networks.utils import eval_mode, train_mode
from monai.transforms import Transform
from monai.utils import ForwardMode, ensure_tuple, min_version, optional_import
from monai.utils.enums import CommonKeys as Keys
from monai.utils.module import look_up_option

if TYPE_CHECKING:
    from ignite.engine import Engine, EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

__all__ = ["Evaluator", "SupervisedEvaluator", "EnsembleEvaluator"]


class Evaluator(Workflow):
    """
    Base class for all kinds of evaluators, inherits from Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable or torch.DataLoader.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
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
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
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

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            max_epochs=1,
            data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            handlers=val_handlers,
            amp=amp,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )
        mode = look_up_option(mode, ForwardMode)
        if mode == ForwardMode.EVAL:
            self.mode = eval_mode
        elif mode == ForwardMode.TRAIN:
            self.mode = train_mode
        else:
            raise ValueError(f"unsupported mode: {mode}, should be 'eval' or 'train'.")

    def run(self, global_epoch: int = 1) -> None:
        """
        Execute validation/evaluation based on Ignite Engine.

        Args:
            global_epoch: the overall epoch if during a training. evaluator engine can get it from trainer.

        """
        # init env value for current validation process
        self.state.max_epochs = global_epoch
        self.state.epoch = global_epoch - 1
        self.state.iteration = 0
        super().run()

    def get_validation_stats(self) -> dict[str, float]:
        return {"best_validation_metric": self.state.best_metric, "best_validation_epoch": self.state.best_metric_epoch}


class SupervisedEvaluator(Evaluator):
    """
    Standard supervised evaluation method with image and label(optional), inherits from evaluator and Workflow.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable, typically be torch.DataLoader.
        network: network to evaluate in the evaluator, should be regular PyTorch `torch.nn.Module`.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
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

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        network: torch.nn.Module,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.network = network
        self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(self, engine: SupervisedEvaluator, batchdata: dict[str, torch.Tensor]):
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - PRED: prediction result of model.

        Args:
            engine: `SupervisedEvaluator` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 2:
            inputs, targets = batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            inputs, targets, args, kwargs = batch

        # put iteration outputs into engine.state
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        # execute forward computation
        with engine.mode(engine.network):

            if engine.amp:
                with torch.cuda.amp.autocast(**engine.amp_kwargs):
                    engine.state.output[Keys.PRED] = engine.inferer(inputs, engine.network, *args, **kwargs)
            else:
                engine.state.output[Keys.PRED] = engine.inferer(inputs, engine.network, *args, **kwargs)
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output


class EnsembleEvaluator(Evaluator):
    """
    Ensemble evaluation for multiple models, inherits from evaluator and Workflow.
    It accepts a list of models for inference and outputs a list of predictions for further operations.

    Args:
        device: an object representing the device on which to run.
        val_data_loader: Ignite engine use data_loader to run, must be Iterable, typically be torch.DataLoader.
        epoch_length: number of iterations for one epoch, default to `len(val_data_loader)`.
        networks: networks to evaluate in order in the evaluator, should be regular PyTorch `torch.nn.Module`.
        pred_keys: the keys to store every prediction data.
            the length must exactly match the number of networks.
            if None, use "pred_{index}" as key corresponding to N networks, index from `0` to `N-1`.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function to parse expected data (usually `image`, `label` and other network args)
            from `engine.state.batch` for every iteration, for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.create_supervised_trainer.html.
        iteration_update: the callable function for every iteration, expect to accept `engine`
            and `engine.state.batch` as inputs, return data will be stored in `engine.state.output`.
            if not provided, use `self._iteration()` instead. for more details please refer to:
            https://pytorch.org/ignite/generated/ignite.engine.engine.Engine.html.
        inferer: inference method that execute model forward on input data, like: SlidingWindow, etc.
        postprocessing: execute additional transformation for the model output data.
            Typically, several Tensor based transforms composed by `Compose`.
        key_val_metric: compute metric when every iteration completed, and save average value to
            engine.state.metrics when epoch completed. key_val_metric is the main metric to compare and save the
            checkpoint into files.
        additional_metrics: more Ignite metrics that also attach to Ignite Engine.
        metric_cmp_fn: function to compare current key metric with previous best key metric value,
            it must accept 2 args (current_metric, previous_best) and return a bool result: if `True`, will update
            `best_metric` and `best_metric_epoch` with current metric and epoch, default to `greater than`.
        val_handlers: every handler is a set of Ignite Event-Handlers, must have `attach` function, like:
            CheckpointHandler, StatsHandler, SegmentationSaver, etc.
        amp: whether to enable auto-mixed-precision evaluation, default is False.
        mode: model forward mode during evaluation, should be 'eval' or 'train',
            which maps to `model.eval()` or `model.train()`, default to 'eval'.
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

    """

    def __init__(
        self,
        device: torch.device,
        val_data_loader: Iterable | DataLoader,
        networks: Sequence[torch.nn.Module],
        pred_keys: KeysCollection | None = None,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        inferer: Inferer | None = None,
        postprocessing: Transform | None = None,
        key_val_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Sequence | None = None,
        amp: bool = False,
        mode: ForwardMode | str = ForwardMode.EVAL,
        event_names: list[str | EventEnum] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        to_kwargs: dict | None = None,
        amp_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp_kwargs=amp_kwargs,
        )

        self.networks = ensure_tuple(networks)
        self.pred_keys = (
            [f"{Keys.PRED}_{i}" for i in range(len(self.networks))] if pred_keys is None else ensure_tuple(pred_keys)
        )
        if len(self.pred_keys) != len(self.networks):
            raise ValueError("length of `pred_keys` must be same as the length of `networks`.")
        self.inferer = SimpleInferer() if inferer is None else inferer

    def _iteration(self, engine: EnsembleEvaluator, batchdata: dict[str, torch.Tensor]):
        """
        callback function for the Supervised Evaluation processing logic of 1 iteration in Ignite Engine.
        Return below items in a dictionary:
            - IMAGE: image Tensor data for model input, already moved to device.
            - LABEL: label Tensor data corresponding to the image, already moved to device.
            - pred_keys[0]: prediction result of network 0.
            - pred_keys[1]: prediction result of network 1.
            - ... ...
            - pred_keys[N]: prediction result of network N.

        Args:
            engine: `EnsembleEvaluator` to execute operation for an iteration.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            ValueError: When ``batchdata`` is None.

        """
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        if len(batch) == 2:
            inputs, targets = batch
            args: tuple = ()
            kwargs: dict = {}
        else:
            inputs, targets, args, kwargs = batch

        # put iteration outputs into engine.state
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

        for idx, network in enumerate(engine.networks):
            with engine.mode(network):
                if engine.amp:
                    with torch.cuda.amp.autocast(**engine.amp_kwargs):
                        if isinstance(engine.state.output, dict):
                            engine.state.output.update(
                                {engine.pred_keys[idx]: engine.inferer(inputs, network, *args, **kwargs)}
                            )
                else:
                    if isinstance(engine.state.output, dict):
                        engine.state.output.update(
                            {engine.pred_keys[idx]: engine.inferer(inputs, network, *args, **kwargs)}
                        )
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output
