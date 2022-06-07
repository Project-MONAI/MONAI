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

import os
import warnings
from contextlib import contextmanager
from typing import Any, Hashable, Mapping, Optional, Tuple

import torch

from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import Transform
from monai.utils.enums import TraceKeys

__all__ = ["TraceableTransform", "InvertibleTransform"]


class TraceableTransform(Transform):
    """
    Maintains a stack of applied transforms. The stack is inserted as pairs of
    `trace_key: list of transforms` to each data dictionary.

    The ``__call__`` method of this transform class must be implemented so
    that the transformation information for each key is stored when
    ``__call__`` is called. If the transforms were applied to keys "image" and
    "label", there will be two extra keys in the dictionary: "image_transforms"
    and "label_transforms" (based on `TraceKeys.KEY_SUFFIX`). Each list
    contains a list of the transforms applied to that key.

    The information in ``data[key_transform]`` will be compatible with the
    default collate since it only stores strings, numbers and arrays.

    `tracing` could be enabled by `self.set_tracing` or setting
    `MONAI_TRACE_TRANSFORM` when initializing the class.
    """

    tracing = not os.environ.get("MONAI_TRACE_TRANSFORM", "1") == "0"

    def set_tracing(self, tracing: bool) -> None:
        """Set whether to trace transforms."""
        self.tracing = tracing

    @staticmethod
    def trace_key(key: Hashable = None):
        """The key to store the stack of applied transforms."""
        if key is None:
            return TraceKeys.KEY_SUFFIX
        return str(key) + TraceKeys.KEY_SUFFIX

    def get_transform_info(
        self, data, key: Hashable = None, extra_info: Optional[dict] = None, orig_size: Optional[Tuple] = None
    ) -> dict:
        """
        Return a dictionary with the relevant information pertaining to an applied
        transform.

        Args:
            - data: input data. Can be dictionary or MetaTensor. We can use `shape` to
                determine the original size of the object (unless that has been given
                explicitly, see `orig_size`).
            - key: if data is a dictionary, data[key] will be modified
            - extra_info: if desired, any extra information pertaining to the applied
                transform can be stored in this dictionary. These are often needed for
                computing the inverse transformation.
            - orig_size: sometimes during the inverse it is useful to know what the size
                of the original image was, in which case it can be supplied here.

        Returns:
            Dictionary of data pertaining to the applied transformation.
        """
        info = {TraceKeys.CLASS_NAME: self.__class__.__name__, TraceKeys.ID: id(self)}
        if orig_size is not None:
            info[TraceKeys.ORIG_SIZE] = orig_size
        elif isinstance(data, Mapping) and key in data and hasattr(data[key], "shape"):
            info[TraceKeys.ORIG_SIZE] = data[key].shape[1:]
        elif hasattr(data, "shape"):
            info[TraceKeys.ORIG_SIZE] = data.shape[1:]
        if extra_info is not None:
            info[TraceKeys.EXTRA_INFO] = extra_info
        # If class is randomizable transform, store whether the transform was actually performed (based on `prob`)
        if hasattr(self, "_do_transform"):  # RandomizableTransform
            info[TraceKeys.DO_TRANSFORM] = self._do_transform  # type: ignore
        return info

    def push_transform(
        self, data, key: Hashable = None, extra_info: Optional[dict] = None, orig_size: Optional[Tuple] = None
    ) -> None:
        """
        Push to a stack of applied transforms.

        Data can be one of two things:
            1. A `MetaTensor`
            2. A dictionary of data containing arrays/tensors and auxilliary data. In
                this case, a key must be supplied.

        If `data` is of type `MetaTensor`, then the applied transform will be added to
            its internal list.

        If `data` is a dictionary, then one of two things can happen:
            1. If data[key] is a `MetaTensor`, the applied transform will be added to
                its internal list.
            2. Else, the applied transform will be appended to an adjacent list using
                `trace_key`. If, for example, the key is `image`, then the transform
                will be appended to `image_transforms`.

        Hopefully it is clear that there are three total possibilities:
            1. data is `MetaTensor`
            2. data is dictionary, data[key] is `MetaTensor`
            3. data is dictionary, data[key] is not `MetaTensor`.

        Args:
            - data: dictionary of data or `MetaTensor`
            - key: if data is a dictionary, data[key] will be modified
            - extra_info: if desired, any extra information pertaining to the applied
                transform can be stored in this dictionary. These are often needed for
                computing the inverse transformation.
            - orig_size: sometimes during the inverse it is useful to know what the size
                of the original image was, in which case it can be supplied here.

        Returns:
            None, but data has been updated to store the applied transformation.
        """
        if not self.tracing:
            return
        info = self.get_transform_info(data, key, extra_info, orig_size)

        if isinstance(data, MetaTensor):
            data.push_applied_operation(info)
        elif isinstance(data, Mapping):
            if key in data and isinstance(data[key], MetaTensor):
                data[key].push_applied_operation(info)
            else:
                # If this is the first, create list
                if self.trace_key(key) not in data:
                    if not isinstance(data, dict):
                        data = dict(data)
                    data[self.trace_key(key)] = []
                data[self.trace_key(key)].append(info)
        else:
            warnings.warn(f"`data` should be either `MetaTensor` or dictionary, {info} not tracked.")

    def check_transforms_match(self, transform: Mapping) -> None:
        """Check transforms are of same instance."""
        xform_id = transform.get(TraceKeys.ID, "")
        if xform_id in [id(self), TraceKeys.NONE]:  # TraceKeys.NONE to skip the check
            return
        xform_name = transform.get(TraceKeys.CLASS_NAME, "")
        # basic check if multiprocessing uses 'spawn' (objects get recreated so don't have same ID)
        if torch.multiprocessing.get_start_method() in ("spawn", None) and xform_name == self.__class__.__name__:
            return
        raise RuntimeError(
            f"Error getting the most recently applied invertible transform {xform_name} {xform_id} != {id(self)}."
        )

    def get_most_recent_transform(self, data, key: Hashable = None, check: bool = True, pop: bool = False):
        """
        Get most recent transform.

        Data can be one of two things:
            1. A `MetaTensor`
            2. A dictionary of data containing arrays/tensors and auxilliary data. In
                this case, a key must be supplied.

        If `data` is of type `MetaTensor`, then the applied transform will be added to
            its internal list.

        If `data` is a dictionary, then one of two things can happen:
            1. If data[key] is a `MetaTensor`, the applied transform will be added to
                its internal list.
            2. Else, the applied transform will be appended to an adjacent list using
                `trace_key`. If, for example, the key is `image`, then the transform
                will be appended to `image_transforms`.

        Hopefully it is clear that there are three total possibilities:
            1. data is `MetaTensor`
            2. data is dictionary, data[key] is `MetaTensor`
            3. data is dictionary, data[key] is not `MetaTensor`.

        Args:
            - data: dictionary of data or `MetaTensor`
            - key: if data is a dictionary, data[key] will be modified
            - check: if true, check that `self` is the same type as the most
                recently-applied transform.
            - pop: if true, remove the transform as it is returned.

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
                all_transforms = data[self.trace_key(key)]
        else:
            raise ValueError(f"`data` should be either `MetaTensor` or dictionary, got {type(data)}.")
        if check:
            self.check_transforms_match(all_transforms[-1])
        return all_transforms.pop() if pop else all_transforms[-1]

    def pop_transform(self, data, key: Hashable = None, check: bool = True):
        """
        Return and pop the most recent transform.

        Data can be one of two things:
            1. A `MetaTensor`
            2. A dictionary of data containing arrays/tensors and auxilliary data. In
                this case, a key must be supplied.

        If `data` is of type `MetaTensor`, then the applied transform will be added to
            its internal list.

        If `data` is a dictionary, then one of two things can happen:
            1. If data[key] is a `MetaTensor`, the applied transform will be added to
                its internal list.
            2. Else, the applied transform will be appended to an adjacent list using
                `trace_key`. If, for example, the key is `image`, then the transform
                will be appended to `image_transforms`.

        Hopefully it is clear that there are three total possibilities:
            1. data is `MetaTensor`
            2. data is dictionary, data[key] is `MetaTensor`
            3. data is dictionary, data[key] is not `MetaTensor`.

        Args:
            - data: dictionary of data or `MetaTensor`
            - key: if data is a dictionary, data[key] will be modified
            - check: if true, check that `self` is the same type as the most
                recently-applied transform.

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

        - the inverse transforms are applied in a last- in-first-out order. As
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

    def inverse(self, data: Any) -> Any:
        """
        Inverse of ``__call__``.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
