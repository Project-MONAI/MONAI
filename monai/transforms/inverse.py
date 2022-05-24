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
from typing import Hashable, Mapping, Optional, Tuple

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

    def push_transform(
        self, data: Mapping, key: Hashable = None, extra_info: Optional[dict] = None, orig_size: Optional[Tuple] = None
    ) -> None:
        """Push to a stack of applied transforms for that key."""

        if not self.tracing:
            return
        info = {TraceKeys.CLASS_NAME: self.__class__.__name__, TraceKeys.ID: id(self)}
        if orig_size is not None:
            info[TraceKeys.ORIG_SIZE] = orig_size
        elif key in data and hasattr(data[key], "shape"):
            info[TraceKeys.ORIG_SIZE] = data[key].shape[1:]
        if extra_info is not None:
            info[TraceKeys.EXTRA_INFO] = extra_info
        # If class is randomizable transform, store whether the transform was actually performed (based on `prob`)
        if hasattr(self, "_do_transform"):  # RandomizableTransform
            info[TraceKeys.DO_TRANSFORM] = self._do_transform  # type: ignore

        if key in data and isinstance(data[key], MetaTensor):
            data[key].push_applied_operation(info)
        else:
            # If this is the first, create list
            if self.trace_key(key) not in data:
                if not isinstance(data, dict):
                    data = dict(data)
                data[self.trace_key(key)] = []
            data[self.trace_key(key)].append(info)

    def pop_transform(self, data: Mapping, key: Hashable = None):
        """Remove the most recent applied transform."""
        if not self.tracing:
            return
        if key in data and isinstance(data[key], MetaTensor):
            return data[key].pop_applied_operation()
        return data.get(self.trace_key(key), []).pop()


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

    def check_transforms_match(self, transform: Mapping) -> None:
        """Check transforms are of same instance."""
        xform_name = transform.get(TraceKeys.CLASS_NAME, "")
        xform_id = transform.get(TraceKeys.ID, "")
        if xform_id == id(self):
            return
        # basic check if multiprocessing uses 'spawn' (objects get recreated so don't have same ID)
        if torch.multiprocessing.get_start_method() in ("spawn", None) and xform_name == self.__class__.__name__:
            return
        raise RuntimeError(f"Error inverting the most recently applied invertible transform {xform_name} {xform_id}.")

    def get_most_recent_transform(self, data: Mapping, key: Hashable = None):
        """Get most recent transform."""
        if not self.tracing:
            raise RuntimeError("Transform Tracing must be enabled to get the most recent transform.")
        if isinstance(data[key], MetaTensor):
            transform = data[key].applied_operations[-1]
        else:
            transform = data[self.trace_key(key)][-1]
        self.check_transforms_match(transform)
        return transform

    def inverse(self, data: dict) -> dict:
        """
        Inverse of ``__call__``.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
