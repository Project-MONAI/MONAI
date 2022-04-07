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

from copy import deepcopy
from typing import Any, Callable, Sequence

_TRACK_META = True
_TRACK_TRANSFORMS = True

__all__ = ["get_track_meta", "get_track_transforms", "set_track_meta", "set_track_transforms", "MetaObj"]


def set_track_meta(val: bool) -> None:
    """
    Boolean to set whether meta data is tracked. If `True`, meta data will be associated
    its data by using subclasses of `MetaObj`. If `False`, then data will be returned
    with empty meta data.

    If both `set_track_meta` and `set_track_transforms` are set to
    `False`, then standard data objects will be returned (e.g., `torch.Tensor` and
    `np.ndarray`) as opposed to our enhanced objects.

    By default, this is `True`, and most users will want to leave it this way. However,
    if you are experiencing any problems regarding meta data, and aren't interested in
    preserving meta data, then you can disable it.
    """
    global _TRACK_META
    _TRACK_META = val


def set_track_transforms(val: bool) -> None:
    """
    Boolean to set whether transforms are tracked. If `True`, applied transforms will be
    associated its data by using subclasses of `MetaObj`. If `False`, then transforms
    won't be tracked.

    If both `set_track_meta` and `set_track_transforms` are set to
    `False`, then standard data objects will be returned (e.g., `torch.Tensor` and
    `np.ndarray`) as opposed to our enhanced objects.

    By default, this is `True`, and most users will want to leave it this way. However,
    if you are experiencing any problems regarding transforms, and aren't interested in
    preserving transforms, then you can disable it.
    """
    global _TRACK_TRANSFORMS
    _TRACK_TRANSFORMS = val


def get_track_meta() -> bool:
    """
    Return the boolean as to whether meta data is tracked. If `True`, meta data will be
    associated its data by using subclasses of `MetaObj`. If `False`, then data will be
    returned with empty meta data.

    If both `set_track_meta` and `set_track_transforms` are set to
    `False`, then standard data objects will be returned (e.g., `torch.Tensor` and
    `np.ndarray`) as opposed to our enhanced objects.

    By default, this is `True`, and most users will want to leave it this way. However,
    if you are experiencing any problems regarding meta data, and aren't interested in
    preserving meta data, then you can disable it.
    """
    return _TRACK_META


def get_track_transforms() -> bool:
    """
    Return the boolean as to whether transforms are tracked. If `True`, applied
    transforms will be associated its data by using subclasses of `MetaObj`. If `False`,
    then transforms won't be tracked.

    If both `set_track_meta` and `set_track_transforms` are set to
    `False`, then standard data objects will be returned (e.g., `torch.Tensor` and
    `np.ndarray`) as opposed to our enhanced objects.

    By default, this is `True`, and most users will want to leave it this way. However,
    if you are experiencing any problems regarding transforms, and aren't interested in
    preserving transforms, then you can disable it.
    """
    return _TRACK_TRANSFORMS


class MetaObj:
    """
    Abstract base class that stores data as well as any extra meta data and an affine
    transformation matrix.

    This allows for subclassing `torch.Tensor` and `np.ndarray` through multiple
    inheritance.

    Meta data is stored in the form of of a dictionary. Affine matrices are stored in
    the form of e.g., `torch.Tensor` or `np.ndarray`.

    We store the affine as its own element, so that this can be updated by
    transforms. All other meta data that we don't plan on touching until we
    need to save the image to file lives in `meta`.

    Behavior should be the same as extended class (e.g., `torch.Tensor` or `np.ndarray`)
    aside from the extended meta functionality.

    Copying of information:
        * For `c = a + b`, then auxiliary data (e.g., meta data) will be copied from the
        first instance of `MetaObj`.
    """

    _meta: dict
    _affine: Any

    def _set_initial_val(self, attribute: str, input_arg: Any, input_obj: Any, default_fn: Callable) -> None:
        """
        Set the initial value of an attribute (e.g., `meta` or `affine`).
        First, try to set `attribute` using `input_arg`. But if `input_arg` is `None`,
        then we try to copy the value from `input_obj`. But if value is also `None`,
        then we finally fall back on using the `default_fn`.

        Args:
            attribute: string corresponding to attribute we want to set (e.g., `meta` or
                `affine`).
            input_arg: the value we would like `attribute` to take or `None` if not
                given.
            input_obj: if `input_arg` is `None`, try to copy `attribute` from
                `input_obj`, if it is present and not `None`.
            default_fn: function to be used if all previous arguments return `None`.
                Default meta data might be empty dictionary so could be as simple as
                `lambda: {}`.
        Returns:
            Returns `None`, but `self` should have the updated `attribute`.
        """
        if input_arg is None:
            input_arg = getattr(input_obj, attribute, None)
        if input_arg is None:
            input_arg = default_fn(self)
        setattr(self, attribute, input_arg)

    @staticmethod
    def flatten_meta_objs(args: Sequence[Any]) -> list[MetaObj]:
        """
        Recursively flatten input and return all instances of `MetaObj` as a single
        list. This means that for both `torch.add(a, b)`, `torch.stack([a, b])` (and
        their numpy equivalents), we return `[a, b]` if both `a` and `b` are of type
        `MetaObj`.

        Args:
            args: Sequence of inputs to be flattened.
        Returns:
            list of nested `MetaObj` from input.
        """
        out = []
        for a in args:
            if isinstance(a, (list, tuple)):
                out += MetaObj.flatten_meta_objs(a)
            elif isinstance(a, MetaObj):
                out.append(a)
        return out

    def _copy_attr(
        self, attribute: str, input_objs: list[MetaObj], default_fn: Callable, deepcopy_required: bool
    ) -> None:
        """
        Copy an attribute from the first in a list of `MetaObj`. In the case of
        `torch.add(a, b)`, both `a` and `b` could be `MetaObj` or something else, so
        check them all. Copy the first to `self`.

        We also perform a deep copy of the data if desired.

        Args:
            attribute: string corresponding to attribute to be copied (e.g., `meta` or
                `affine`).
            input_objs: List of `MetaObj`. We'll copy the attribute from the first one
                that contains that particular attribute.
            default_fn: If none of `input_objs` have the attribute that we're
                interested in, then use this default function (e.g., `lambda: {}`.)
            deepcopy_required: Should the attribute be deep copied? See `_copy_meta`.

        Returns:
            Returns `None`, but `self` should be updated to have the copied attribute.
        """
        attributes = [getattr(i, attribute) for i in input_objs]
        if len(attributes) > 0:
            val = attributes[0]
            if deepcopy_required:
                val = deepcopy(val)
            setattr(self, attribute, val)
        else:
            setattr(self, attribute, default_fn())

    def _copy_meta(self, input_objs: list[MetaObj]) -> None:
        """
        Copy meta data from a list of `MetaObj`. For a given attribute, we copy the
        adjunct data from the first element in the list containing that attribute.

        If there has been a change in `id` (e.g., `a=b+c`), then deepcopy. Else (e.g.,
        `a+=1`), then don't.

        Args:
            input_objs: list of `MetaObj` to copy data from.

        """
        id_in = id(input_objs[0]) if len(input_objs) > 0 else None
        deepcopy_required = id(self) != id_in
        attributes = ("affine", "meta")
        default_fns: tuple[Callable, ...] = (self.get_default_affine, self.get_default_meta)
        for attribute, default_fn in zip(attributes, default_fns):
            self._copy_attr(attribute, input_objs, default_fn, deepcopy_required)

    def get_default_meta(self) -> dict:
        """Get the default meta.

        Returns:
            default meta data.
        """
        return {}

    def get_default_affine(self) -> Any:
        """Get the default affine.

        Returns:
            default affine.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """String representation of class."""
        out: str = super().__repr__()

        out += f"\nAffine\n{self.affine}"

        out += "\nMetaData\n"
        if self.meta is not None:
            out += "".join(f"\t{k}: {v}\n" for k, v in self.meta.items())
        else:
            out += "None"

        return out

    @property
    def affine(self) -> Any:
        """Get the affine."""
        return self._affine

    @affine.setter
    def affine(self, d: Any) -> None:
        """Set the affine."""
        self._affine = d

    @property
    def meta(self) -> dict:
        """Get the meta."""
        return self._meta

    @meta.setter
    def meta(self, d: dict) -> None:
        """Set the meta."""
        self._meta = d
