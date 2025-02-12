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

import copy
import os
import pickle
import tempfile
from collections.abc import Hashable
from types import ModuleType
from typing import Any

import torch
from torch.serialization import DEFAULT_PROTOCOL

from monai.config.type_definitions import PathLike

__all__ = ["StateCacher"]


class StateCacher:
    """Class to cache and retrieve the state of an object.

    Objects can either be stored in memory or on disk. If stored on disk, they can be
    stored in a given directory, or alternatively a temporary location will be used.

    If necessary/possible, restored objects will be returned to their original device.

    Example:

    >>> state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
    >>> state_cacher.store("model", model.state_dict())
    >>> model.load_state_dict(state_cacher.retrieve("model"))
    """

    def __init__(
        self,
        in_memory: bool,
        cache_dir: PathLike | None = None,
        allow_overwrite: bool = True,
        pickle_module: ModuleType = pickle,
        pickle_protocol: int = DEFAULT_PROTOCOL,
    ) -> None:
        """Constructor.

        Args:
            in_memory: boolean to determine if the object will be cached in memory or on
                disk.
            cache_dir: directory for data to be cached if `in_memory==False`. Defaults
                to using a temporary directory. Any created files will be deleted during
                the `StateCacher`'s destructor.
            allow_overwrite: allow the cache to be overwritten. If set to `False`, an
                error will be thrown if a matching already exists in the list of cached
                objects.
            pickle_module: module used for pickling metadata and objects, default to `pickle`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.
            pickle_protocol: can be specified to override the default protocol, default to `2`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.

        """
        self.in_memory = in_memory
        self.cache_dir = tempfile.gettempdir() if cache_dir is None else cache_dir
        if not os.path.isdir(self.cache_dir):
            raise ValueError("Given `cache_dir` is not a valid directory.")

        self.allow_overwrite = allow_overwrite
        self.pickle_module = pickle_module
        self.pickle_protocol = pickle_protocol
        self.cached: dict = {}

    def store(
        self, key: Hashable, data_obj: Any, pickle_module: ModuleType | None = None, pickle_protocol: int | None = None
    ) -> None:
        """
        Store a given object with the given key name.

        Args:
            key: key of the data object to store.
            data_obj: data object to store.
            pickle_module: module used for pickling metadata and objects, default to `self.pickle_module`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.
            pickle_protocol: can be specified to override the default protocol, default to `self.pickle_protocol`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.

        """
        if key in self.cached and not self.allow_overwrite:
            raise RuntimeError("Cached key already exists and overwriting is disabled.")
        if self.in_memory:
            self.cached.update({key: {"obj": copy.deepcopy(data_obj)}})
        else:
            fn = os.path.join(self.cache_dir, f"state_{key}_{id(self)}.pt")
            self.cached.update({key: {"obj": fn}})
            torch.save(
                obj=data_obj,
                f=fn,
                pickle_module=self.pickle_module if pickle_module is None else pickle_module,
                pickle_protocol=self.pickle_protocol if pickle_protocol is None else pickle_protocol,
            )
            # store object's device if relevant
            if hasattr(data_obj, "device"):
                self.cached[key]["device"] = data_obj.device

    def retrieve(self, key: Hashable) -> Any:
        """Retrieve the object stored under a given key name."""
        if key not in self.cached:
            raise KeyError(f"Target {key} was not cached.")

        if self.in_memory:
            return self.cached[key]["obj"]

        fn = self.cached[key]["obj"]  # pytype: disable=attribute-error
        if not os.path.exists(fn):  # pytype: disable=wrong-arg-types
            raise RuntimeError(f"Failed to load state in {fn}. File doesn't exist anymore.")
        data_obj = torch.load(fn, map_location=lambda storage, location: storage)
        # copy back to device if necessary
        if "device" in self.cached[key]:
            data_obj = data_obj.to(self.cached[key]["device"])
        return data_obj

    def __del__(self):
        """If necessary, delete any cached files existing in `cache_dir`."""
        if not self.in_memory:
            for k in self.cached:
                if os.path.exists(self.cached[k]["obj"]):
                    os.remove(self.cached[k]["obj"])
