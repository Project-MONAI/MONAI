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

import os
import warnings
from collections.abc import Hashable, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from monai import transforms
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import LazyTransform, Transform
from monai.utils.enums import LazyAttr, TraceKeys
from monai.utils.type_conversion import convert_to_numpy, convert_to_tensor

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

    tracing = os.environ.get("MONAI_TRACE_TRANSFORM", "1") != "0"

    def set_tracing(self, tracing: bool) -> None:
        """Set whether to trace transforms."""
        self.tracing = tracing

    @staticmethod
    def trace_key(key: Hashable = None):
        """The key to store the stack of applied transforms."""
        if key is None:
            return f"{TraceKeys.KEY_SUFFIX}"
        return f"{key}{TraceKeys.KEY_SUFFIX}"

    def get_transform_info(self) -> dict:
        """
        Return a dictionary with the relevant information pertaining to an applied transform.
        """
        return {
            TraceKeys.CLASS_NAME: self.__class__.__name__,
            TraceKeys.ID: id(self),
            TraceKeys.TRACING: self.tracing,
            TraceKeys.LAZY_EVALUATION: self.lazy_evaluation if isinstance(self, LazyTransform) else False,
            # If class is randomizable transform, store whether the transform was actually performed (based on `prob`)
            TraceKeys.DO_TRANSFORM: self._do_transform if hasattr(self, "_do_transform") else False,
        }

    def push_transform(self, data, *args, **kwargs):
        transform_info = self.get_transform_info()
        lazy_eval = transform_info.get(TraceKeys.LAZY_EVALUATION, False)
        do_transform = transform_info.get(TraceKeys.DO_TRANSFORM, False)
        if not kwargs:
            kwargs = {}
        kwargs["transform_info"] = transform_info
        replace = kwargs.pop("replace", False)
        if replace and isinstance(data, MetaTensor) and get_track_meta():
            if not lazy_eval:
                xform = self.pop_transform(data, check=False) if do_transform else {}
                return self.push_transform(data, extra_info=xform)
            elif do_transform:
                return self.push_transform(data, pending=data.pending_operations.pop())  # type: ignore
            else:
                return data
        if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
            return TraceableTransform.track_pending_transform(data, *args, **kwargs)
        return TraceableTransform.track_transform(data, *args, **kwargs)

    @classmethod
    def track_transform(
        cls,
        data,
        key: Hashable = None,
        extra_info: dict | None = None,
        orig_size: tuple | None = None,
        transform_info=None,
    ):
        """
        Push to a stack of applied transforms.

        Args:
            data: dictionary of data or `MetaTensor`.
            key: if data is a dictionary, data[key] will be modified.
            extra_info: if desired, any extra information pertaining to the applied
                transform can be stored in this dictionary. These are often needed for
                computing the inverse transformation.
            orig_size: sometimes during the inverse it is useful to know what the size
                of the original image was, in which case it can be supplied here.
            transform_info: the information pertaining to the applied transform.

        Returns:
            None, but data has been updated to store the applied transformation.
        """
        if not get_track_meta() or not transform_info or not transform_info.get(TraceKeys.TRACING):
            return data
        info = transform_info
        if orig_size is not None:
            info[TraceKeys.ORIG_SIZE] = orig_size
        elif isinstance(data, Mapping) and key in data and isinstance(data[key], MetaTensor):
            info[TraceKeys.ORIG_SIZE] = data[key].peek_pending_shape()
        elif isinstance(data, Mapping) and key in data and hasattr(data[key], "shape"):
            info[TraceKeys.ORIG_SIZE] = data[key].shape[1:]
        elif isinstance(data, MetaTensor):
            info[TraceKeys.ORIG_SIZE] = data.peek_pending_shape()
        elif hasattr(data, "shape"):
            info[TraceKeys.ORIG_SIZE] = data.shape[1:]
        if extra_info is not None:
            info[TraceKeys.EXTRA_INFO] = extra_info

        if isinstance(data, MetaTensor):
            data.push_applied_operation(info)
        elif isinstance(data, Mapping):
            if key in data and isinstance(data[key], MetaTensor):
                data[key].push_applied_operation(info)
            else:
                # If this is the first, create list
                if TraceableTransform.trace_key(key) not in data:
                    if not isinstance(data, dict):
                        data = dict(data)
                    data[TraceableTransform.trace_key(key)] = []
                data[TraceableTransform.trace_key(key)].append(info)
        else:
            warnings.warn(f"`data` should be either `MetaTensor` or dictionary, got {type(data)}. {info} not tracked.")
        return data

    @classmethod
    def track_pending_transform(
        cls,
        data,
        key: Hashable = None,
        lazy_shape=None,
        lazy_affine=None,
        extra_info: dict | None = None,
        orig_size: tuple | None = None,
        pending=None,
        transform_info=None,
    ):
        """
        Push to MetaTensor's pending operations for later execution.

        See also: `track_transform`.
        """
        if not get_track_meta() or not transform_info or not transform_info.get(TraceKeys.TRACING):
            return data
        info = transform_info
        if orig_size is not None:
            info[TraceKeys.ORIG_SIZE] = orig_size
        elif isinstance(data, Mapping) and key in data and isinstance(data[key], MetaTensor):
            info[TraceKeys.ORIG_SIZE] = data[key].peek_pending_shape()
        elif isinstance(data, Mapping) and key in data and hasattr(data[key], "shape"):
            info[TraceKeys.ORIG_SIZE] = data[key].shape[1:]
        elif isinstance(data, MetaTensor):
            info[TraceKeys.ORIG_SIZE] = data.peek_pending_shape()
        elif hasattr(data, "shape"):
            info[TraceKeys.ORIG_SIZE] = data.shape[1:]
        if extra_info is not None:
            info[TraceKeys.EXTRA_INFO] = extra_info

        if pending is not None:
            pending.pop(TraceKeys.CLASS_NAME, None)
            pending.pop(TraceKeys.ID, None)
            pending.pop(TraceKeys.DO_TRANSFORM, None)
            pending.pop(TraceKeys.TRACING, None)
            pending.pop(TraceKeys.LAZY_EVALUATION, None)
            info.update(pending)
        if lazy_shape is not None:
            info[LazyAttr.SHAPE] = tuple(convert_to_numpy(lazy_shape, wrap_sequence=True).tolist())
        if lazy_affine is not None:
            info[LazyAttr.AFFINE] = convert_to_tensor(lazy_affine, device=torch.device("cpu"))
        if isinstance(data, MetaTensor):
            data.push_pending_operation(info)
        elif isinstance(data, Mapping) and key in data and isinstance(data[key], MetaTensor):
            data[key].push_pending_operation(info)
        else:
            warnings.warn(f"`data` should be either `MetaTensor` or dictionary, got {type(data)}. {info} not tracked.")
        return data

    def check_transforms_match(self, transform: Mapping) -> None:
        """Check transforms are of same instance."""
        xform_id = transform.get(TraceKeys.ID, "")
        if xform_id == id(self):
            return
        # TraceKeys.NONE to skip the id check
        if xform_id == TraceKeys.NONE:
            return
        xform_name = transform.get(TraceKeys.CLASS_NAME, "")
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


class InvertibleTransform(TraceableTransform):
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
