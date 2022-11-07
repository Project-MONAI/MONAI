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

from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.optimizer import Optimizer

from monai.config import IgniteInfo
from monai.engines.utils import get_devices_spec
from monai.utils import min_version, optional_import

create_supervised_trainer, _ = optional_import(
    "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "create_supervised_trainer"
)
create_supervised_evaluator, _ = optional_import(
    "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "create_supervised_evaluator"
)
_prepare_batch, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "_prepare_batch")
if TYPE_CHECKING:
    from ignite.engine import Engine
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric", as_type="decorator")

__all__ = ["create_multigpu_supervised_trainer", "create_multigpu_supervised_evaluator"]


def _default_transform(_x: torch.Tensor, _y: torch.Tensor, _y_pred: torch.Tensor, loss: torch.Tensor) -> float:
    return loss.item()


def _default_eval_transform(
    x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return y_pred, y


def create_multigpu_supervised_trainer(
    net: torch.nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    devices: Optional[Sequence[torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = _default_transform,
    distributed: bool = False,
):
    """
    Derived from `create_supervised_trainer` in Ignite.

    Factory function for creating a trainer for supervised models.

    Args:
        net: the network to train.
        optimizer: the optimizer to use.
        loss_fn: the loss function to use.
        devices: device(s) type specification (default: None).
            Applies to both model and batches. None is all devices used, empty list is CPU only.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        distributed: whether convert model to `DistributedDataParallel`, if `True`, `devices` must contain
            only 1 GPU or CPU for current distributed rank.

    Returns:
        Engine: a trainer engine with supervised update function.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.
    """

    devices_ = get_devices_spec(devices)
    if distributed:
        if len(devices_) > 1:
            raise ValueError(f"for distributed training, `devices` must contain only 1 GPU or CPU, but got {devices_}.")
        net = DistributedDataParallel(net, device_ids=devices_)
    elif len(devices_) > 1:
        net = DataParallel(net)

    return create_supervised_trainer(
        net, optimizer, loss_fn, devices_[0], non_blocking, prepare_batch, output_transform
    )


def create_multigpu_supervised_evaluator(
    net: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    devices: Optional[Sequence[torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = _default_eval_transform,
    distributed: bool = False,
):
    """
    Derived from `create_supervised_evaluator` in Ignite.

    Factory function for creating an evaluator for supervised models.

    Args:
        net: the model to train.
        metrics: a map of metric names to Metrics.
        devices: device(s) type specification (default: None).
            Applies to both model and batches. None is all devices used, empty list is CPU only.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)`
            which fits output expected by metrics. If you change it you should use `output_transform` in metrics.
        distributed: whether convert model to `DistributedDataParallel`, if `True`, `devices` must contain
            only 1 GPU or CPU for current distributed rank.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """

    devices_ = get_devices_spec(devices)

    if distributed:
        net = DistributedDataParallel(net, device_ids=devices_)
        if len(devices_) > 1:
            raise ValueError(
                f"for distributed evaluation, `devices` must contain only 1 GPU or CPU, but got {devices_}."
            )
    elif len(devices_) > 1:
        net = DataParallel(net)

    return create_supervised_evaluator(net, metrics, devices_[0], non_blocking, prepare_batch, output_transform)
