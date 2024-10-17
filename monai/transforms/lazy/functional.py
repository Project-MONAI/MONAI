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

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from monai.apps.utils import get_logger
from monai.config import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.transforms.lazy.utils import (
    affine_from_pending,
    combine_transforms,
    is_compatible_apply_kwargs,
    kwargs_from_pending,
    resample,
)
from monai.transforms.traits import LazyTrait
from monai.transforms.transform import MapTransform
from monai.utils import LazyAttr, look_up_option

__all__ = ["apply_pending_transforms", "apply_pending_transforms_in_order", "apply_pending"]

__override_keywords = {"mode", "padding_mode", "dtype", "align_corners", "resample_mode", "device"}


def _log_pending_info(
    transform: Any,
    data: Any,
    activity: str,
    *,
    lazy: bool | None = None,
    key: str | None = None,
    logger_name: bool | str = False,
):
    if logger_name is False:
        return
    logger_name = logger_name if isinstance(logger_name, str) else "apply_pending_transforms"
    logger = get_logger(logger_name)

    tcname = type(transform).__name__
    if isinstance(transform, LazyTrait):
        tlazy = f", transform.lazy: {transform.lazy}"
        if lazy is not None and lazy != transform.lazy:
            tlazy += " (overridden)"
    else:
        tlazy = ", transform is not lazy"

    msg = f"{activity} - lazy: {lazy}, {{key_msg}}pending: {{pcount}}, upcoming '{tcname}'{tlazy}"

    if isinstance(transform, MapTransform):
        transform_keys = transform.keys if key is None else (key,)
        for k in transform_keys:
            if k in data:
                pcount = len(data[k].pending_operations) if isinstance(data[k], MetaTensor) else 0
                logger.info(msg.format(pcount=pcount, key_msg=f"key: '{k}', "))
    else:
        pcount = len(data.pending_operations) if isinstance(data, MetaTensor) else 0
        logger.info(msg.format(pcount=pcount, key_msg="" if key is None else f"key: '{key}', "))


def _log_applied_info(data: Any, key=None, logger_name: bool | str = False):
    if logger_name is False:
        return
    logger_name = logger_name if isinstance(logger_name, str) else "apply_pending_transforms"
    logger = get_logger(logger_name)

    key_str = "" if key is None else f"key: '{key}', "
    logger.info(f"Pending transforms applied: {key_str}applied_operations: {len(data.applied_operations)}")


def apply_pending_transforms(
    data: NdarrayOrTensor | Sequence[Any | NdarrayOrTensor] | Mapping[Any, NdarrayOrTensor],
    keys: tuple | None,
    overrides: dict | None = None,
    logger_name: bool | str = False,
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
    if isinstance(data, list):
        return [apply_pending_transforms(d, keys, overrides, logger_name) for d in data]
    if isinstance(data, tuple):
        return tuple(apply_pending_transforms(d, keys, overrides, logger_name) for d in data)

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
    transform, data, lazy: bool | None = None, overrides: dict | None = None, logger_name: bool | str = False
):
    """
    This method causes "in order" processing of pending transforms to occur.
    "in order" processing of pending transforms ensures that all pending transforms have been applied to the
    tensor before a non-lazy transform (or lazy transform that is executing non-lazily) is carried out.
    It ensures that no operations will be added to a metatensor's apply_operations while there are outstanding
    pending_operations. Note that there is only one mechanism for executing lazy resampling at present but this
    is expected to change in future releases.

    Evaluation of pending transforms is performed under the following circumstances:
    * If the transform is a lazy transform and:
      * The transform checks data as part of its execution, or
      * the transform is not executing lazily
    * If the transform is an ApplyPending[d] transform
    * If the transform is not a lazy transform

    This method is designed to be used only in the context of implementing lazy resampling functionality. In general
    you should not need to interact with or use this method directly, and its API may change without warning between
    releases. See the :ref:`Lazy Resampling topic<lazy_resampling> for more information about lazy resampling.

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
    from monai.transforms.lazy.dictionary import ApplyPendingd

    must_apply_pending = True
    keys = transform.keys if isinstance(transform, ApplyPendingd) else None
    if isinstance(transform, LazyTrait) and not transform.requires_current_data:
        must_apply_pending = not (transform.lazy if lazy is None else lazy)

    if must_apply_pending is True:
        _log_pending_info(transform, data, "Apply pending transforms", lazy=lazy, logger_name=logger_name)
        return apply_pending_transforms(data, keys, overrides, logger_name)

    _log_pending_info(transform, data, "Accumulate pending transforms", lazy=lazy, logger_name=logger_name)
    return data


def apply_pending(data: torch.Tensor | MetaTensor, pending: list | None = None, overrides: dict | None = None):
    """
    This method applies pending transforms to `data` tensors.
    Currently, only 2d and 3d inputs are supported.

    This method is designed to be called by ``apply_pending_transforms`` and other methods / classes
    that are part of the implementation of lazy resampling. In general, you should not need to call
    this method unless you are directly developing custom lazy execution strategies.

    It works by calculating the overall effect of the accumulated pending transforms. When it runs
    out of pending transforms or when it finds incompatibilities between the accumulated pending
    transform and the next pending transform, it then applies the accumulated transform in a call to
    ``resample``.

    Pending transforms are incompatible with each other if one or more of the arguments in the pending
    transforms differ. These are parameters such as 'mode', 'padding_mode', 'dtype' and so forth. If
    a pending transform doesn't have a given parameter, it is considered compatible with the
    accumulated transform. If a subsequent transform has a parameter that is incompatible with
    the accumulated transform (e.g. 'mode' of 'bilinear' vs. 'mode' of 'nearest'), an intermediate
    resample will be performed and the accumulated transform reset to its starting state.

    After resampling, the pending transforms are pushed to the ``applied_transforms`` field of the
    resulting MetaTensor. Note, if a torch.tensor is passed to this method along with a list of
    pending transforms, the resampled tensor will be wrapped in a MetaTensor before being returned.

    Args:
        data: A torch Tensor or a monai MetaTensor.
        pending: pending transforms. This must be set if data is a Tensor, but is optional if data is a MetaTensor.
        overrides: a dictionary of overrides for the transform arguments. The keys must be one of:

            - mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order ``0-5`` (integers).
                Interpolation mode to calculate output values. Defaults to None.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's `an integer`, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            - padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to None.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            - dtype: data type for resampling computation. Defaults to ``float64``.
                If ``None``, use the data type of input data, this option may not be compatible the resampling backend.
            - align_corners: Geometrically, we consider the pixels of the input as squares rather than points, when using
                the PyTorch resampling backend. Defaults to ``False``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            - device: device for resampling computation. Defaults to ``None``.
            - resample_mode: the mode of resampling, currently support ``"auto"``. Setting to other values will use the
                :py:class:`monai.transforms.SpatialResample` for resampling (instead of potentially crop/pad).
    """
    overrides = (overrides or {}).copy()
    for k in overrides:
        look_up_option(k, __override_keywords)  # check existence of the key

    if isinstance(data, MetaTensor) and pending is None:
        pending = data.pending_operations.copy()
        data.clear_pending_operations()
    pending = [] if pending is None else pending

    if not pending:
        return data, []

    cumulative_xform = affine_from_pending(pending[0])
    if cumulative_xform.shape[0] == 3:
        cumulative_xform = to_affine_nd(3, cumulative_xform)

    cur_kwargs = kwargs_from_pending(pending[0])
    override_kwargs: dict[str, Any] = {}
    if "mode" in overrides:
        override_kwargs[LazyAttr.INTERP_MODE] = overrides["mode"]
    if "padding_mode" in overrides:
        override_kwargs[LazyAttr.PADDING_MODE] = overrides["padding_mode"]
    if "align_corners" in overrides:
        override_kwargs[LazyAttr.ALIGN_CORNERS] = overrides["align_corners"]
    if "resample_mode" in overrides:
        override_kwargs[LazyAttr.RESAMPLE_MODE] = overrides["resample_mode"]
    override_dtype = overrides.get("dtype", torch.float64)
    override_kwargs[LazyAttr.DTYPE] = data.dtype if override_dtype is None else override_dtype
    device = overrides.get("device")

    for p in pending[1:]:
        new_kwargs = kwargs_from_pending(p)
        if not is_compatible_apply_kwargs(cur_kwargs, new_kwargs):
            # carry out an intermediate resample here due to incompatibility between arguments
            _cur_kwargs = cur_kwargs.copy()
            _cur_kwargs.update(override_kwargs)
            data = resample(data.to(device), cumulative_xform, _cur_kwargs)

        next_matrix = affine_from_pending(p)
        if next_matrix.shape[0] == 3:
            next_matrix = to_affine_nd(3, next_matrix)

        cumulative_xform = combine_transforms(cumulative_xform, next_matrix)
        cur_kwargs.update(new_kwargs)
    cur_kwargs.update(override_kwargs)
    data = resample(data.to(device), cumulative_xform, cur_kwargs)
    if isinstance(data, MetaTensor):
        for p in pending:
            data.push_applied_operation(p)
    return data, pending
