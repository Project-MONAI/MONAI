# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Hashable, Optional, Tuple

import numpy as np
import torch

from monai.transforms.transform import RandomizableTransform, Transform
from monai.utils.enums import InverseKeys

__all__ = ["InvertibleTransform"]


class InvertibleTransform(Transform):
    """Classes for invertible transforms.

    This class exists so that an ``invert`` method can be implemented. This allows, for
    example, images to be cropped, rotated, padded, etc., during training and inference,
    and after be returned to their original size before saving to file for comparison in
    an external viewer.

    When the ``__call__`` method is called, the transformation information for each key is
    stored. If the transforms were applied to keys "image" and "label", there will be two
    extra keys in the dictionary: "image_transforms" and "label_transforms". Each list
    contains a list of the transforms applied to that key. When the ``inverse`` method is
    called, the inverse is called on each key individually, which allows for different
    parameters being passed to each label (e.g., different interpolation for image and
    label).

    When the ``inverse`` method is called, the inverse transforms are applied in a last-
    in-first-out order. As the inverse is applied, its entry is removed from the list
    detailing the applied transformations. That is to say that during the forward pass,
    the list of applied transforms grows, and then during the inverse it shrinks back
    down to an empty list.

    The information in ``data[key_transform]`` will be compatible with the default collate
    since it only stores strings, numbers and arrays.

    We currently check that the ``id()`` of the transform is the same in the forward and
    inverse directions. This is a useful check to ensure that the inverses are being
    processed in the correct order. However, this may cause issues if the ``id()`` of the
    object changes (such as multiprocessing on Windows). If you feel this issue affects
    you, please raise a GitHub issue.

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

    def push_transform(
        self,
        data: dict,
        key: Hashable,
        extra_info: Optional[dict] = None,
        orig_size: Optional[Tuple] = None,
    ) -> None:
        """Append to list of applied transforms for that key."""
        key_transform = str(key) + InverseKeys.KEY_SUFFIX
        info = {
            InverseKeys.CLASS_NAME: self.__class__.__name__,
            InverseKeys.ID: id(self),
            InverseKeys.ORIG_SIZE: orig_size or (data[key].shape[1:] if hasattr(data[key], "shape") else None),
        }
        if extra_info is not None:
            info[InverseKeys.EXTRA_INFO] = extra_info
        # If class is randomizable transform, store whether the transform was actually performed (based on `prob`)
        if isinstance(self, RandomizableTransform):
            info[InverseKeys.DO_TRANSFORM] = self._do_transform
        # If this is the first, create list
        if key_transform not in data:
            data[key_transform] = []
        data[key_transform].append(info)

    def check_transforms_match(self, transform: dict) -> None:
        """Check transforms are of same instance."""
        if transform[InverseKeys.ID] == id(self):
            return
        # basic check if multiprocessing uses 'spawn' (objects get recreated so don't have same ID)
        if (
            torch.multiprocessing.get_start_method(allow_none=False) == "spawn"
            and transform[InverseKeys.CLASS_NAME] == self.__class__.__name__
        ):
            return
        raise RuntimeError("Should inverse most recently applied invertible transform first")

    def get_most_recent_transform(self, data: dict, key: Hashable) -> dict:
        """Get most recent transform."""
        transform = dict(data[str(key) + InverseKeys.KEY_SUFFIX][-1])
        self.check_transforms_match(transform)
        return transform

    def pop_transform(self, data: dict, key: Hashable) -> None:
        """Remove most recent transform."""
        data[str(key) + InverseKeys.KEY_SUFFIX].pop()

    def inverse(self, data: dict) -> Dict[Hashable, np.ndarray]:
        """
        Inverse of ``__call__``.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
