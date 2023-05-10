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

from typing import Any, Mapping

from monai.apps.utils import get_logger
from monai.config import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.transforms.lazy.array import ApplyPending
from monai.transforms.lazy.dictionary import ApplyPendingd
from monai.transforms.lazy.functional import apply_pending
from monai.transforms.traits import LazyTrait
from monai.transforms.transform import MapTransform

__all__ = ["apply_pending_transforms", "apply_pending_transforms_in_order", "apply_pending_transforms_out_of_order"]


def _log_pending_info(
    transform: Any,
    data: Any,
    activity: str,
    *,
    lazy: bool | None = None,
    key: str | None = None,
    logger_name: str | None = None,
):
    if logger_name is None:
        return
    logger = get_logger(logger_name)

    if isinstance(transform, LazyTrait):
        if lazy is not None and lazy != transform.lazy:
            tlazy = f", transform.lazy: {transform.lazy} (overridden)"
        else:
            tlazy = f", transform.lazy: {transform.lazy}"
    else:
        tlazy = ", transform is not lazy"

    if isinstance(transform, MapTransform):
        transform_keys = transform.keys if key is None else (key,)
        for k in transform_keys:
            if k in data:
                pcount = len(data[k].pending_operations) if isinstance(data[k], MetaTensor) else 0
                logger.info(
                    f"{activity} - lazy mode: {lazy}, key: '{k}', "
                    f"pending: {pcount}, upcoming '{transform.__class__.__name__}'{tlazy}"
                )
    else:
        pcount = len(data.pending_operations) if isinstance(data, MetaTensor) else 0
        if key is None:
            logger.info(
                f"{activity} - lazy: {lazy}, " f"pending: {pcount}, upcoming '{transform.__class__.__name__}'{tlazy}"
            )
        else:
            logger.info(
                f"{activity} - lazy mode: {lazy}, key: '{key}', "
                f"pending: {pcount}, upcoming '{transform.__class__.__name__}'{tlazy}"
            )


def _log_applied_info(data: Any, key=None, logger_name: str | None = None):
    if logger_name is None:
        return
    logger = get_logger(logger_name)

    key_str = "" if key is None else f"key: '{key}', "
    logger.info(f"Pending transforms applied: {key_str}applied_operations: {len(data.applied_operations)}")


def apply_pending_transforms(
    data: NdarrayOrTensor | Mapping[Any, NdarrayOrTensor],
    keys: tuple | None,
    overrides: dict | None = None,
    logger_name: str | None = None,
):
    """
    apply_pending_transforms is called with either a tensor or a dictionary, some entries of which contain
    tensors.

    When operating on a dictionary of tensors, the 'keys' parameter determines what tensors should be checked.
    If 'keys' is not set, all keys of 'data' are considered.

    This method optionally takes a set of overrides that can be used to change specific parameters on the
    transform pipeline. See ``Compose`` for more details. This method takes a logger_name that can be used
    to override the default logger, to provide telemetry during the execution of pending transforms.

    This method is intended primarily for use by ``execute_compose`` and other methods that handle the
    underlying execution of transform pipelines. You should not need to use it in the general case, unless
    you are developing functionality to perform such operations.

    Args:
        data: a ``torch.Tensor`` or ``MetaTensor``, or dictionary of tensors.
        keys: an optional tuple of keys that filters the keys on 'data' if it is a dict
        overrides: An optional dictionary that specifies parameters that can be used to override transform
            arguments when they are called. When 'data' is a dict, this dictionary should contain a dictionary
            of overrides for each key that needs them
        logger_name: An optional name for a logger to be used when applying pending transforms. If None,
            logging is suppressed.
    Returns:
        an object of the same type as data if pending transforms were applied, or 'data' if they were not
    """
    if isinstance(data, dict):
        # get the keys from 'data' for metatensors with pending operations. If 'keys' is set, select
        # only data keys that are in 'keys'
        active_keys = [k for k in data.keys() if keys is None or k in keys]
        keys_to_update = [k for k in active_keys if isinstance(data[k], MetaTensor) and data[k].has_pending_operations]

        if len(keys_to_update) > 0:
            rdata = dict(data)

            for k in keys_to_update:
                overrides_ = None if overrides is None else overrides.get(k, None)
                rdata[k], _ = apply_pending(data[k], overrides=overrides_)
                _log_applied_info(rdata[k], key=k, logger_name=logger_name)

            return rdata
    else:
        if isinstance(data, MetaTensor) and data.has_pending_operations:
            rdata, _ = apply_pending(data, overrides=overrides)
            _log_applied_info(rdata, logger_name=logger_name)
            return rdata

    return data


def apply_pending_transforms_in_order(
    transform, data, lazy: bool | None = None, overrides: dict | None = None, logger_name: str | None = None
):
    """
    This method causes "out of order" processing of pending transforms to occur.

    Out of order processing for lazy resampling only causes pending transforms to be processed when
    an `ApplyPending`_ or `ApplyPendingd`_ transform is encountered in the pipeline.

    This method is designed to be used only in the context of implementing lazy resampling functionality. In general
    you should not need to interact with or use this method directly.
    Args:
        transform: a transform that should be evaluated to determine whether pending transforms should be applied
        data: a tensor / MetaTensor, or dictionary containing tensors / MetaTensors whose pending transforms may
            need to be applied
        lazy: The lazy mode that is being applied (this can be False, True or None)
        overrides: An optional dictionary containing overrides to be applied to the pending transforms when they
            are lazily executed. If data is a dict, it should contain a dictionary of overrides for each key that
            needs them
        logger_name: An optional name for a logger to be used when applying pending transforms. If None,
            logging is suppressed.
    Returns:
        an object of the same type as data if pending transforms were applied, or 'data' if they were not

    """
    apply_pending = False
    keys = None
    if isinstance(transform, LazyTrait):
        if transform.partially_lazy:
            apply_pending = True
        else:
            apply_pending = not (transform.lazy if lazy is None else lazy)
    elif isinstance(transform, ApplyPending):
        apply_pending = True
    elif isinstance(transform, ApplyPendingd):
        apply_pending = True
        keys = transform.keys
    else:
        apply_pending = True

    if apply_pending is True:
        _log_pending_info(transform, data, "Apply pending transforms", lazy=lazy, logger_name=logger_name)
        return apply_pending_transforms(data, keys, overrides, logger_name)

    _log_pending_info(transform, data, "Accumulate pending transforms", lazy=lazy, logger_name=logger_name)
    return data


def apply_pending_transforms_out_of_order(
    transform, data, lazy: bool | None = None, overrides: dict | None = None, logger_name: str | None = None
):
    """
    This method causes "out of order" processing of pending transforms to occur.

    Out of order processing for lazy resampling only causes pending transforms to be processed when
    an `ApplyPending`_ or `ApplyPendingd`_ transform is encountered in the pipeline.

    This method is designed to be used only in the context of implementing lazy resampling functionality. In general
    you should not need to interact with or use this method directly.
    Args:
        transform: a transform that should be evaluated to determine whether pending transforms should be applied
        data: a tensor / MetaTensor, or dictionary containing tensors / MetaTensors whose pending transforms may
            need to be applied
        lazy: The lazy mode that is being applied (this can be False, True or None)
        overrides: An optional dictionary containing overrides to be applied to the pending transforms when they
            are lazily executed. If data is a dict, it should contain a dictionary of overrides for each key that
            needs them
        logger_name: An optional name for a logger to be used when applying pending transforms. If None,
            logging is suppressed.
    Returns:
        an object of the same type as data if pending transforms were applied, or 'data' if they were not

    """
    if lazy is False:
        _log_pending_info(transform, data, "Apply pending transforms", lazy=lazy, logger_name=logger_name)
        return apply_pending_transforms(data, None, overrides, logger_name)

    if isinstance(transform, ApplyPendingd):
        _log_pending_info(transform, data, "Apply pending transforms", lazy=lazy, logger_name=logger_name)
        return apply_pending_transforms(data, transform.keys, overrides, logger_name)

    if isinstance(transform, ApplyPending):
        _log_pending_info(transform, data, "Apply pending transforms", lazy=lazy, logger_name=logger_name)
        return apply_pending_transforms(data, None, overrides, logger_name)

    _log_pending_info(transform, data, "Accumulate pending transforms", lazy=lazy, logger_name=logger_name)

    return data
