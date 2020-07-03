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

from typing import Callable
import torch

from monai.utils import exact_version, optional_import
from monai.engines.utils import get_devices_spec

create_supervised_trainer, _ = optional_import("ignite.engine", "0.3.0", exact_version, "create_supervised_trainer")
create_supervised_evaluator, _ = optional_import("ignite.engine", "0.3.0", exact_version, "create_supervised_evaluator")
_prepare_batch, _ = optional_import("ignite.engine", "0.3.0", exact_version, "_prepare_batch")


def _default_transform(_x, _y, _y_pred, loss):
    return loss.item()


def _default_eval_transform(x, y, y_pred):
    return y_pred, y


def create_multigpu_supervised_trainer(
    net: torch.nn.Module,
    optimizer,
    loss_fn,
    devices=None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = _default_transform,
):
    """
    Derived from `create_supervised_trainer` in Ignite.

    Factory function for creating a trainer for supervised models.

    Args:
        net (`torch.nn.Module`): the network to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (`torch.nn` loss function): the loss function to use.
        devices (list, optional): device(s) type specification (default: None).
            Applies to both model and batches. None is all devices used, empty list is CPU only.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Returns:
        Engine: a trainer engine with supervised update function.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.
    """

    devices = get_devices_spec(devices)

    if len(devices) > 1:
        net = torch.nn.parallel.DataParallel(net)

    return create_supervised_trainer(net, optimizer, loss_fn, devices[0], non_blocking, prepare_batch, output_transform)


def create_multigpu_supervised_evaluator(
    net: torch.nn.Module,
    metrics=None,
    devices=None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    output_transform: Callable = _default_eval_transform,
):
    """
    Derived from `create_supervised_evaluator` in Ignite.

    Factory function for creating an evaluator for supervised models.

    Args:
        net (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of metric names to Metrics.
        devices (list, optional): device(s) type specification (default: None).
            Applies to both model and batches. None is all devices used, empty list is CPU only.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    Returns:
        Engine: an evaluator engine with supervised inference function.
    """

    devices = get_devices_spec(devices)

    if len(devices) > 1:
        net = torch.nn.parallel.DataParallel(net)

    return create_supervised_evaluator(net, metrics, devices[0], non_blocking, prepare_batch, output_transform)
