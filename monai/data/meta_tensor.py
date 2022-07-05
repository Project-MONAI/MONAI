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
from copy import deepcopy
from typing import Any, Sequence

import torch

from monai.config.type_definitions import NdarrayTensor
from monai.data.meta_obj import MetaObj, get_track_meta
from monai.data.utils import affine_to_spacing, decollate_batch, list_data_collate, remove_extra_metadata
from monai.utils.enums import PostFix
from monai.utils.type_conversion import convert_to_tensor

__all__ = ["MetaTensor"]


class MetaTensor(MetaObj, torch.Tensor):
    """
    Class that inherits from both `torch.Tensor` and `MetaObj`, adding support for metadata.

    Metadata is stored in the form of a dictionary. Nested, an affine matrix will be
    stored. This should be in the form of `torch.Tensor`.

    Behavior should be the same as `torch.Tensor` aside from the extended
    meta functionality.

    Copying of information:

        * For `c = a + b`, then auxiliary data (e.g., metadata) will be copied from the
          first instance of `MetaTensor` if `a.is_batch` is False
          (For batched data, the metdata will be shallow copied for efficiency purposes).

    Example:
        .. code-block:: python

            import torch
            from monai.data import MetaTensor

            t = torch.tensor([1,2,3])
            affine = torch.as_tensor([[2,0,0,0],
                                      [0,2,0,0],
                                      [0,0,2,0],
                                      [0,0,0,1]], dtype=torch.float64)
            meta = {"some": "info"}
            m = MetaTensor(t, affine=affine, meta=meta)
            m2 = m + m
            assert isinstance(m2, MetaTensor)
            assert m2.meta["some"] == "info"
            assert torch.all(m2.affine == affine)

    Notes:
        - Requires pytorch 1.9 or newer for full compatibility.
        - Older versions of pytorch (<=1.8), `torch.jit.trace(net, im)` may
          not work if `im` is of type `MetaTensor`. This can be resolved with
          `torch.jit.trace(net, im.as_tensor())`.
        - For pytorch < 1.8, sharing `MetaTensor` instances across processes may not be supported.
        - A warning will be raised if in the constructor `affine` is not `None` and
          `meta` already contains the key `affine`.
        - You can query whether the `MetaTensor` is a batch with the `is_batch` attribute.
        - With a batch of data, `batch[0]` will return the 0th image
          with the 0th metadata. When the batch dimension is non-singleton, e.g.,
          `batch[:, 0]`, `batch[..., -1]` and `batch[1:3]`, then all (or a subset in the
          last example) of the metadata will be returned, and `is_batch` will return `True`.
        - When creating a batch with this class, use `monai.data.DataLoader` as opposed
          to `torch.utils.data.DataLoader`, as this will take care of collating the
          metadata properly.
    """

    @staticmethod
    def __new__(
        cls,
        x,
        affine: torch.Tensor | None = None,
        meta: dict | None = None,
        applied_operations: list | None = None,
        *args,
        **kwargs,
    ) -> MetaTensor:
        _kwargs = {"device": kwargs.pop("device", None), "dtype": kwargs.pop("dtype", None)} if kwargs else {}
        return torch.as_tensor(x, *args, **_kwargs).as_subclass(cls)  # type: ignore

    def __init__(
        self,
        x,
        affine: torch.Tensor | None = None,
        meta: dict | None = None,
        applied_operations: list | None = None,
        *_args,
        **_kwargs,
    ) -> None:
        """
        If `meta` is given, use it. Else, if `meta` exists in the input tensor, use it.
        Else, use the default value. Similar for the affine, except this could come from
        four places.
        Priority: `affine`, `meta["affine"]`, `x.affine`, `get_default_affine`.
        """
        super().__init__()
        # set meta
        if meta is not None:
            self.meta = meta
        elif isinstance(x, MetaObj):
            self.meta = x.meta
        # set the affine
        if affine is not None:
            if "affine" in self.meta:
                warnings.warn("Setting affine, but the applied meta contains an affine. This will be overwritten.")
            self.affine = affine
        elif "affine" in self.meta:
            # by using the setter function, we ensure it is converted to torch.Tensor if not already
            self.affine = self.meta["affine"]
        elif isinstance(x, MetaTensor):
            self.affine = x.affine
        else:
            self.affine = self.get_default_affine()
        # applied_operations
        if applied_operations is not None:
            self.applied_operations = applied_operations
        elif isinstance(x, MetaTensor):
            self.applied_operations = x.applied_operations
        else:
            self.applied_operations = MetaObj.get_default_applied_operations()

        # if we are creating a new MetaTensor, then deep copy attributes
        if isinstance(x, torch.Tensor) and not isinstance(x, MetaTensor):
            self.meta = deepcopy(self.meta)
            self.applied_operations = deepcopy(self.applied_operations)
        self.affine = self.affine.to(self.device)

    def _copy_attr(self, attributes: list[str], input_objs, defaults: list, deep_copy: bool) -> None:
        super()._copy_attr(attributes, input_objs, defaults, deep_copy)
        for a in attributes:
            val = getattr(self, a)
            if isinstance(val, torch.Tensor):
                setattr(self, a, val.to(self.device))

    @staticmethod
    def update_meta(rets: Sequence, func, args, kwargs) -> Sequence:
        """
        Update the metadata from the output of `MetaTensor.__torch_function__`.

        The output of `torch.Tensor.__torch_function__` could be a single object or a
        sequence of them. Hence, in `MetaTensor.__torch_function__` we convert them to a
        list of not already, and then we loop across each element, processing metadata
        as necessary. For each element, if not of type `MetaTensor`, then nothing to do.

        Args:
            rets: the output from `torch.Tensor.__torch_function__`, which has been
                converted to a list in `MetaTensor.__torch_function__` if it wasn't
                already a `Sequence`.
            func: the torch function that was applied. Examples might be `torch.squeeze`
                or `torch.Tensor.__add__`. We need this since the metadata need to be
                treated differently if a batch of data is considered. For example,
                slicing (`torch.Tensor.__getitem__`) the ith element of the 0th
                dimension of a batch of data should return a ith tensor with the ith
                metadata.
            args: positional arguments that were passed to `func`.
            kwargs: keyword arguments that were passed to `func`.

        Returns:
            A sequence with the same number of elements as `rets`. For each element, if
            the input type was not `MetaTensor`, then no modifications will have been
            made. If global parameters have been set to false (e.g.,
            `not get_track_meta()`), then any `MetaTensor` will be converted to
            `torch.Tensor`. Else, metadata will be propogated as necessary (see
            :py:func:`MetaTensor._copy_meta`).
        """
        out = []
        metas = None
        is_batch = any(x.is_batch for x in MetaObj.flatten_meta_objs(args, kwargs.values()) if hasattr(x, "is_batch"))
        for idx, ret in enumerate(rets):
            # if not `MetaTensor`, nothing to do.
            if not isinstance(ret, MetaTensor):
                pass
            # if not tracking, convert to `torch.Tensor`.
            elif not get_track_meta():
                ret = ret.as_tensor()
            # else, handle the `MetaTensor` metadata.
            else:
                meta_args = MetaObj.flatten_meta_objs(args, kwargs.values())
                ret._copy_meta(meta_args, deep_copy=not is_batch)
                ret.is_batch = is_batch
                # the following is not implemented but the network arch may run into this case:
                # if func == torch.cat and any(m.is_batch if hasattr(m, "is_batch") else False for m in meta_args):
                #     raise NotImplementedError("torch.cat is not implemented for batch of MetaTensors.")

                # If we have a batch of data, then we need to be careful if a slice of
                # the data is returned. Depending on how the data are indexed, we return
                # some or all of the metadata, and the return object may or may not be a
                # batch of data (e.g., `batch[:,-1]` versus `batch[0]`).
                if is_batch:
                    # if indexing e.g., `batch[0]`
                    if func == torch.Tensor.__getitem__:
                        batch_idx = args[1]
                        if isinstance(batch_idx, Sequence):
                            batch_idx = batch_idx[0]
                        # if using e.g., `batch[:, -1]` or `batch[..., -1]`, then the
                        # first element will be `slice(None, None, None)` and `Ellipsis`,
                        # respectively. Don't need to do anything with the metadata.
                        if batch_idx not in (slice(None, None, None), Ellipsis, None):
                            # only decollate metadata once
                            if metas is None:
                                metas = decollate_batch(ret.meta)
                            meta = metas[batch_idx]
                            # if using e.g., `batch[0:2]`, then `is_batch` should still be
                            # `True`. Also re-collate the remaining elements.
                            if isinstance(meta, list):
                                ret.meta = list_data_collate(meta)
                            # if using e.g., `batch[0]` or `batch[0, 1]`, then return single
                            # element from batch, and set `is_batch` to `False`.
                            else:
                                ret.meta = meta
                                ret.is_batch = False
                    # `unbind` is used for `next(iter(batch))`. Also for `decollate_batch`.
                    # But we only want to split the batch if the `unbind` is along the 0th
                    # dimension.
                    elif func == torch.Tensor.unbind:
                        if len(args) > 1:
                            dim = args[1]
                        elif "dim" in kwargs:
                            dim = kwargs["dim"]
                        else:
                            dim = 0
                        if dim == 0:
                            if metas is None:
                                metas = decollate_batch(ret.meta)
                            ret.meta = metas[idx]
                            ret.is_batch = False

                ret.affine = ret.affine.to(ret.device)
            out.append(ret)
        # if the input was a tuple, then return it as a tuple
        return tuple(out) if isinstance(rets, tuple) else out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None) -> Any:
        """Wraps all torch functions."""
        if kwargs is None:
            kwargs = {}
        ret = super().__torch_function__(func, types, args, kwargs)
        # if `out` has been used as argument, metadata is not copied, nothing to do.
        # if "out" in kwargs:
        #     return ret
        # we might have 1 or multiple outputs. Might be MetaTensor, might be something
        # else (e.g., `__repr__` returns a string).
        # Convert to list (if necessary), process, and at end remove list if one was added.
        if (
            hasattr(torch, "return_types")
            and hasattr(func, "__name__")
            and hasattr(torch.return_types, func.__name__)
            and isinstance(getattr(torch.return_types, func.__name__), type)
            and isinstance(ret, getattr(torch.return_types, func.__name__))
        ):
            # for torch.max(torch.tensor(1.0), dim=0), the return type is named-tuple like
            out_items = MetaTensor.update_meta(ret, func, args, kwargs)
            for idx in range(ret.n_fields):
                ret[idx].meta = out_items[idx].meta
                ret[idx].applied_operations = out_items[idx].applied_operations
            return ret
        if isinstance(ret, (str, bytes)) or not isinstance(ret, Sequence):
            ret = [ret]
            unpack = True
        else:
            unpack = False
        ret = MetaTensor.update_meta(ret, func, args, kwargs)
        return ret[0] if unpack else ret

    def get_default_affine(self, dtype=torch.float64) -> torch.Tensor:
        return torch.eye(4, device=self.device, dtype=dtype)

    def as_tensor(self) -> torch.Tensor:
        """
        Return the `MetaTensor` as a `torch.Tensor`.
        It is OS dependent as to whether this will be a deep copy or not.
        """
        return self.as_subclass(torch.Tensor)  # type: ignore

    def as_dict(self, key: str) -> dict:
        """
        Get the object as a dictionary for backwards compatibility.
        This method makes a copy of the objects.

        Args:
            key: Base key to store main data. The key for the metadata will be
                determined using `PostFix.meta`.

        Return:
            A dictionary consisting of two keys, the main data (stored under `key`) and
                the metadata.
        """
        return {
            key: self.as_tensor().clone().detach(),
            PostFix.meta(key): deepcopy(self.meta),
            PostFix.transforms(key): deepcopy(self.applied_operations),
        }

    @property
    def affine(self) -> torch.Tensor:
        """Get the affine."""
        return self.meta.get("affine", self.get_default_affine())

    @affine.setter
    def affine(self, d: NdarrayTensor) -> None:
        """Set the affine."""
        self.meta["affine"] = torch.as_tensor(d, device=self.device)

    @property
    def pixdim(self):
        """Get the spacing"""
        return affine_to_spacing(self.affine)

    def new_empty(self, size, dtype=None, device=None, requires_grad=False):
        """
        must be defined for deepcopy to work

        See:
            - https://pytorch.org/docs/stable/generated/torch.Tensor.new_empty.html#torch-tensor-new-empty
        """
        return type(self)(
            self.as_tensor().new_empty(size=size, dtype=dtype, device=device, requires_grad=requires_grad)
        )

    @staticmethod
    def ensure_torch_and_prune_meta(im: NdarrayTensor, meta: dict):
        """
        Convert the image to `torch.Tensor`. If `affine` is in the `meta` dictionary,
        convert that to `torch.Tensor`, too. Remove any superfluous metadata.

        Args:
            im: Input image (`np.ndarray` or `torch.Tensor`)
            meta: Metadata dictionary.

        Returns:
            By default, a `MetaTensor` is returned.
            However, if `get_track_meta()` is `False`, a `torch.Tensor` is returned.
        """
        img = convert_to_tensor(im)  # potentially ascontiguousarray

        # if not tracking metadata, return `torch.Tensor`
        if not get_track_meta() or meta is None:
            return img

        # ensure affine is of type `torch.Tensor`
        if "affine" in meta:
            meta["affine"] = convert_to_tensor(meta["affine"])

        # remove any superfluous metadata.
        remove_extra_metadata(meta)

        # return the `MetaTensor`
        return MetaTensor(img, meta=meta)

    def __repr__(self, *, tensor_contents=None):
        return self.as_tensor().__repr__() + super().__repr__()
