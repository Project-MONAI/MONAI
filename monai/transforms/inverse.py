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
from collections.abc import Hashable, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from monai import transforms
from monai.data.meta_obj import MetaObj, get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.transforms.traits import InvertibleTrait
from monai.transforms.transform import Transform
from monai.utils import (
    LazyAttr,
    MetaKeys,
    TraceKeys,
    TraceStatusKeys,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
)
from monai.utils.misc import MONAIEnvVars

__all__ = ["TraceableTransform", "InvertibleTransform"]


class TraceableTransform(Transform):
    """
    Maintains a stack of applied transforms to data.

    Data can be one of two types:
        1. A `MetaTensor` (this is the preferred data type).
        2. A dictionary of data containing arrays/tensors and auxiliary metadata. In
            this case, a key must be supplied (this dictionary-based approach is deprecated).

    If `data` is of type `MetaTensor`, then the applied transform will be added to ``data.applied_operations``.

    If `data` is a dictionary, then one of two things can happen:
        1. If data[key] is a `MetaTensor`, the applied transform will be added to ``data[key].applied_operations``.
        2. Else, the applied transform will be appended to an adjacent list using
            `trace_key`. If, for example, the key is `image`, then the transform
            will be appended to `image_transforms` (this dictionary-based approach is deprecated).

    Hopefully it is clear that there are three total possibilities:
        1. data is `MetaTensor`
        2. data is dictionary, data[key] is `MetaTensor`
        3. data is dictionary, data[key] is not `MetaTensor` (this is a deprecated approach).

    The ``__call__`` method of this transform class must be implemented so
    that the transformation information is stored during the data transformation.

    The information in the stack of applied transforms must be compatible with the
    default collate, by only storing strings, numbers and arrays.

    `tracing` could be enabled by `self.set_tracing` or setting
    `MONAI_TRACE_TRANSFORM` when initializing the class.
    """

    tracing = MONAIEnvVars.trace_transform() != "0"

    def set_tracing(self, tracing: bool) -> None:
        """Set whether to trace transforms."""
        self.tracing = tracing

    @staticmethod
    def trace_key(key: Hashable = None):
        """The key to store the stack of applied transforms."""
        if key is None:
            return f"{TraceKeys.KEY_SUFFIX}"
        return f"{key}{TraceKeys.KEY_SUFFIX}"

    @staticmethod
    def transform_info_keys():
        """The keys to store necessary info of an applied transform."""
        return (TraceKeys.CLASS_NAME, TraceKeys.ID, TraceKeys.TRACING, TraceKeys.DO_TRANSFORM)

    def get_transform_info(self) -> dict:
        """
        Return a dictionary with the relevant information pertaining to an applied transform.
        """
        vals = (
            self.__class__.__name__,
            id(self),
            self.tracing,
            self._do_transform if hasattr(self, "_do_transform") else True,
        )
        return dict(zip(self.transform_info_keys(), vals))

    def push_transform(self, data, *args, **kwargs):
        """
        Push to a stack of applied transforms of ``data``.

        Args:
            data: dictionary of data or `MetaTensor`.
            args: additional positional arguments to track_transform_meta.
            kwargs: additional keyword arguments to track_transform_meta,
                set ``replace=True`` (default False) to rewrite the last transform infor in
                applied_operation/pending_operation based on ``self.get_transform_info()``.
        """
        lazy_eval = kwargs.get("lazy", False)
        transform_info = self.get_transform_info()
        do_transform = transform_info.get(TraceKeys.DO_TRANSFORM, True)
        kwargs = kwargs or {}
        replace = kwargs.pop("replace", False)  # whether to rewrite the most recently pushed transform info
        if replace and get_track_meta() and isinstance(data, MetaTensor):
            if not lazy_eval:
                xform = self.pop_transform(data, check=False) if do_transform else {}
                meta_obj = self.push_transform(data, orig_size=xform.get(TraceKeys.ORIG_SIZE), extra_info=xform)
                return data.copy_meta_from(meta_obj)
            if do_transform:
                xform = data.pending_operations.pop()
                extra = xform.copy()
                xform.update(transform_info)
            else:  # lazy, replace=True, do_transform=False
                xform, extra = transform_info, {}
            meta_obj = self.push_transform(data, transform_info=xform, lazy=True, extra_info=extra)
            return data.copy_meta_from(meta_obj)
        kwargs["lazy"] = lazy_eval
        if "transform_info" in kwargs and isinstance(kwargs["transform_info"], dict):
            kwargs["transform_info"].update(transform_info)
        else:
            kwargs["transform_info"] = transform_info
        meta_obj = TraceableTransform.track_transform_meta(data, *args, **kwargs)
        return data.copy_meta_from(meta_obj) if isinstance(data, MetaTensor) else data

    @classmethod
    def track_transform_meta(
        cls,
        data,
        key: Hashable = None,
        sp_size=None,
        affine=None,
        extra_info: dict | None = None,
        orig_size: tuple | None = None,
        transform_info=None,
        lazy=False,
    ):
        """
        Update a stack of applied/pending transforms metadata of ``data``.

        Args:
            data: dictionary of data or `MetaTensor`.
            key: if data is a dictionary, data[key] will be modified.
            sp_size: the expected output spatial size when the transform is applied.
                it can be tensor or numpy, but will be converted to a list of integers.
            affine: the affine representation of the (spatial) transform in the image space.
                When the transform is applied, meta_tensor.affine will be updated to ``meta_tensor.affine @ affine``.
            extra_info: if desired, any extra information pertaining to the applied
                transform can be stored in this dictionary. These are often needed for
                computing the inverse transformation.
            orig_size: sometimes during the inverse it is useful to know what the size
                of the original image was, in which case it can be supplied here.
            transform_info: info from self.get_transform_info().
            lazy: whether to push the transform to pending_operations or applied_operations.

        Returns:

            For backward compatibility, if ``data`` is a dictionary, it returns the dictionary with
            updated ``data[key]``. Otherwise, this function returns a MetaObj with updated transform metadata.
        """
        data_t = data[key] if key is not None else data  # compatible with the dict data representation
        out_obj = MetaObj()
        # after deprecating metadict, we should always convert data_t to metatensor here
        if isinstance(data_t, MetaTensor):
            out_obj.copy_meta_from(data_t, keys=out_obj.__dict__.keys())

        if lazy and (not get_track_meta()):
            warnings.warn("metadata is not tracked, please call 'set_track_meta(True)' if doing lazy evaluation.")

        if not lazy and affine is not None and isinstance(data_t, MetaTensor):
            # not lazy evaluation, directly update the metatensor affine (don't push to the stack)
            orig_affine = data_t.peek_pending_affine()
            orig_affine = convert_to_dst_type(orig_affine, affine, dtype=torch.float64)[0]
            try:
                affine = orig_affine @ to_affine_nd(len(orig_affine) - 1, affine, dtype=torch.float64)
            except RuntimeError as e:
                if orig_affine.ndim > 2:
                    if data_t.is_batch:
                        msg = "Transform applied to batched tensor, should be applied to instances only"
                    else:
                        msg = "Mismatch affine matrix, ensured that the batch dimension is not included in the calculation."
                    raise RuntimeError(msg) from e
                else:
                    raise
            out_obj.meta[MetaKeys.AFFINE] = convert_to_tensor(affine, device=torch.device("cpu"), dtype=torch.float64)

        if not (get_track_meta() and transform_info and transform_info.get(TraceKeys.TRACING)):
            if isinstance(data, Mapping):
                if not isinstance(data, dict):
                    data = dict(data)
                data[key] = data_t.copy_meta_from(out_obj) if isinstance(data_t, MetaTensor) else data_t
                return data
            return out_obj  # return with data_t as tensor if get_track_meta() is False

        info = transform_info.copy()
        # track the current spatial shape
        if orig_size is not None:
            info[TraceKeys.ORIG_SIZE] = orig_size
        elif isinstance(data_t, MetaTensor):
            info[TraceKeys.ORIG_SIZE] = data_t.peek_pending_shape()
        elif hasattr(data_t, "shape"):
            info[TraceKeys.ORIG_SIZE] = data_t.shape[1:]

        # add lazy status to the transform info
        info[TraceKeys.LAZY] = lazy

        # include extra_info
        if extra_info is not None:
            extra_info.pop(LazyAttr.SHAPE, None)
            extra_info.pop(LazyAttr.AFFINE, None)
            info[TraceKeys.EXTRA_INFO] = extra_info

        # push the transform info to the applied_operation or pending_operation stack
        if lazy:
            if sp_size is None:
                if LazyAttr.SHAPE not in info:
                    info[LazyAttr.SHAPE] = info.get(TraceKeys.ORIG_SIZE, [])
            else:
                info[LazyAttr.SHAPE] = sp_size
            info[LazyAttr.SHAPE] = tuple(convert_to_numpy(info[LazyAttr.SHAPE], wrap_sequence=True).tolist())
            if affine is None:
                if LazyAttr.AFFINE not in info:
                    info[LazyAttr.AFFINE] = MetaTensor.get_default_affine()
            else:
                info[LazyAttr.AFFINE] = affine
            info[LazyAttr.AFFINE] = convert_to_tensor(info[LazyAttr.AFFINE], device=torch.device("cpu"))
            out_obj.push_pending_operation(info)
        else:
            if out_obj.pending_operations:
                transform_name = info.get(TraceKeys.CLASS_NAME, "") if isinstance(info, dict) else ""
                msg = (
                    f"Transform {transform_name} has been applied to a MetaTensor with pending operations: "
                    f"{[x.get(TraceKeys.CLASS_NAME) for x in out_obj.pending_operations]}"
                )
                if key is not None:
                    msg += f" for key {key}"

                pend = out_obj.pending_operations[-1]
                statuses = pend.get(TraceKeys.STATUSES, dict())
                messages = statuses.get(TraceStatusKeys.PENDING_DURING_APPLY, list())
                messages.append(msg)
                statuses[TraceStatusKeys.PENDING_DURING_APPLY] = messages
                info[TraceKeys.STATUSES] = statuses
            out_obj.push_applied_operation(info)
        if isinstance(data, Mapping):
            if not isinstance(data, dict):
                data = dict(data)
            if isinstance(data_t, MetaTensor):
                data[key] = data_t.copy_meta_from(out_obj)
            else:
                x_k = TraceableTransform.trace_key(key)
                if x_k not in data:
                    data[x_k] = []  # If this is the first, create list
                data[x_k].append(info)
            return data
        return out_obj

    def check_transforms_match(self, transform: Mapping) -> None:
        """Check transforms are of same instance."""
        xform_id = transform.get(TraceKeys.ID, "")
        if xform_id == id(self):
            return
        # TraceKeys.NONE to skip the id check
        if xform_id == TraceKeys.NONE:
            return
        xform_name = transform.get(TraceKeys.CLASS_NAME, "")
        warning_msg = transform.get(TraceKeys.EXTRA_INFO, {}).get("warn")
        if warning_msg:
            warnings.warn(warning_msg)
        # basic check if multiprocessing uses 'spawn' (objects get recreated so don't have same ID)
        if torch.multiprocessing.get_start_method() in ("spawn", None) and xform_name == self.__class__.__name__:
            return
        raise RuntimeError(
            f"Error {self.__class__.__name__} getting the most recently "
            f"applied invertible transform {xform_name} {xform_id} != {id(self)}."
        )

    def get_most_recent_transform(self, data, key: Hashable = None, check: bool = True, pop: bool = False):
        """
        Get most recent transform for the stack.

        Args:
            data: dictionary of data or `MetaTensor`.
            key: if data is a dictionary, data[key] will be modified.
            check: if true, check that `self` is the same type as the most recently-applied transform.
            pop: if true, remove the transform as it is returned.

        Returns:
            Dictionary of most recently applied transform

        Raises:
            - RuntimeError: data is neither `MetaTensor` nor dictionary
        """
        if not self.tracing:
            raise RuntimeError("Transform Tracing must be enabled to get the most recent transform.")
        if isinstance(data, MetaTensor):
            all_transforms = data.applied_operations
        elif isinstance(data, Mapping):
            if key in data and isinstance(data[key], MetaTensor):
                all_transforms = data[key].applied_operations
            else:
                all_transforms = data.get(self.trace_key(key), MetaTensor.get_default_applied_operations())
        else:
            raise ValueError(f"`data` should be either `MetaTensor` or dictionary, got {type(data)}.")
        if check:
            self.check_transforms_match(all_transforms[-1])
        return all_transforms.pop() if pop else all_transforms[-1]

    def pop_transform(self, data, key: Hashable = None, check: bool = True):
        """
        Return and pop the most recent transform.

        Args:
            data: dictionary of data or `MetaTensor`
            key: if data is a dictionary, data[key] will be modified
            check: if true, check that `self` is the same type as the most recently-applied transform.

        Returns:
            Dictionary of most recently applied transform

        Raises:
            - RuntimeError: data is neither `MetaTensor` nor dictionary
        """
        return self.get_most_recent_transform(data, key, check, pop=True)

    @contextmanager
    def trace_transform(self, to_trace: bool):
        """Temporarily set the tracing status of a transform with a context manager."""
        prev = self.tracing
        self.tracing = to_trace
        yield
        self.tracing = prev


class InvertibleTransform(TraceableTransform, InvertibleTrait):
    """Classes for invertible transforms.

    This class exists so that an ``invert`` method can be implemented. This allows, for
    example, images to be cropped, rotated, padded, etc., during training and inference,
    and after be returned to their original size before saving to file for comparison in
    an external viewer.

    When the ``inverse`` method is called:

        - the inverse is called on each key individually, which allows for
          different parameters being passed to each label (e.g., different
          interpolation for image and label).

        - the inverse transforms are applied in a last-in-first-out order. As
          the inverse is applied, its entry is removed from the list detailing
          the applied transformations. That is to say that during the forward
          pass, the list of applied transforms grows, and then during the
          inverse it shrinks back down to an empty list.

    We currently check that the ``id()`` of the transform is the same in the forward and
    inverse directions. This is a useful check to ensure that the inverses are being
    processed in the correct order.

    Note to developers: When converting a transform to an invertible transform, you need to:

        #. Inherit from this class.
        #. In ``__call__``, add a call to ``push_transform``.
        #. Any extra information that might be needed for the inverse can be included with the
           dictionary ``extra_info``. This dictionary should have the same keys regardless of
           whether ``do_transform`` was `True` or `False` and can only contain objects that are
           accepted in pytorch data loader's collate function (e.g., `None` is not allowed).
        #. Implement an ``inverse`` method. Make sure that after performing the inverse,
           ``pop_transform`` is called.

    """

    def inverse_update(self, data):
        """
        This function is to be called before every `self.inverse(data)`,
        update each MetaTensor `data[key]` using `data[key_transforms]` and `data[key_meta_dict]`,
        for MetaTensor backward compatibility 0.9.0.
        """
        if not isinstance(data, dict) or not isinstance(self, transforms.MapTransform):
            return data
        d = dict(data)
        for k in self.key_iterator(data):
            transform_key = transforms.TraceableTransform.trace_key(k)
            if transform_key not in data or not data[transform_key]:
                continue
            d = transforms.sync_meta_info(k, data, t=False)
        return d

    def inverse(self, data: Any) -> Any:
        """
        Inverse of ``__call__``.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
