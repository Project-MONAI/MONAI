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

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.multiprocessing import Manager
from torch.serialization import DEFAULT_PROTOCOL
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset

from monai.data.utils import SUPPORTED_PICKLE_MOD, convert_tables_to_dicts, pickle_hashing
from monai.transforms import (
    Compose,
    Randomizable,
    ThreadUnsafe,
    Transform,
    apply_transform,
    convert_to_contiguous,
    reset_ops_id,
)
from monai.utils import MAX_SEED, deprecated_arg, get_seed, look_up_option, min_version, optional_import
from monai.utils.misc import first

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")


class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)


class DatasetFunc(Dataset):
    """
    Execute function on the input dataset and leverage the output to act as a new Dataset.
    It can be used to load / fetch the basic dataset items, like the list of `image, label` paths.
    Or chain together to execute more complicated logic, like `partition_dataset`, `resample_datalist`, etc.
    The `data` arg of `Dataset` will be applied to the first arg of callable `func`.
    Usage example::

        data_list = DatasetFunc(
            data="path to file",
            func=monai.data.load_decathlon_datalist,
            data_list_key="validation",
            base_dir="path to base dir",
        )
        # partition dataset for every rank
        data_partition = DatasetFunc(
            data=data_list,
            func=lambda **kwargs: monai.data.partition_dataset(**kwargs)[torch.distributed.get_rank()],
            num_partitions=torch.distributed.get_world_size(),
        )
        dataset = Dataset(data=data_partition, transform=transforms)

    Args:
        data: input data for the func to process, will apply to `func` as the first arg.
        func: callable function to generate dataset items.
        kwargs: other arguments for the `func` except for the first arg.

    """

    def __init__(self, data: Any, func: Callable, **kwargs) -> None:
        super().__init__(data=None, transform=None)  # type:ignore
        self.src = data
        self.func = func
        self.kwargs = kwargs
        self.reset()

    def reset(self, data: Optional[Any] = None, func: Optional[Callable] = None, **kwargs):
        """
        Reset the dataset items with specified `func`.

        Args:
            data: if not None, execute `func` on it, default to `self.src`.
            func: if not None, execute the `func` with specified `kwargs`, default to `self.func`.
            kwargs: other arguments for the `func` except for the first arg.

        """
        src = self.src if data is None else data
        self.data = self.func(src, **self.kwargs) if func is None else func(src, **kwargs)


class PersistentDataset(Dataset):
    """
    Persistent storage of pre-computed values to efficiently manage larger than memory dictionary format data,
    it can operate transforms for specific fields.  Results from the non-random transform components are computed
    when first used, and stored in the `cache_dir` for rapid retrieval on subsequent uses.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    The transforms which are supposed to be cached must implement the `monai.transforms.Transform`
    interface and should not be `Randomizable`. This dataset will cache the outcomes before the first
    `Randomizable` `Transform` within a `Compose` instance.

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
            'image': 'image1.nii.gz',    'image': 'image2.nii.gz',    'image': 'image3.nii.gz',
            'label': 'label1.nii.gz',    'label': 'label2.nii.gz',    'label': 'label3.nii.gz',
            'extra': 123                 'extra': 456                 'extra': 789
        },                           },                           }]

    For a composite transform like

    .. code-block:: python

        [ LoadImaged(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96),
                                pos=1, neg=1, num_samples=4, image_key='image', image_threshold=0),
        ToTensord(keys=['image', 'label'])]

    Upon first use a filename based dataset will be processed by the transform for the
    [LoadImaged, Orientationd, ScaleIntensityRanged] and the resulting tensor written to
    the `cache_dir` before applying the remaining random dependant transforms
    [RandCropByPosNegLabeld, ToTensord] elements for use in the analysis.

    Subsequent uses of a dataset directly read pre-processed results from `cache_dir`
    followed by applying the random dependant parts of transform processing.

    During training call `set_data()` to update input data and recompute cache content.

    Note:
        The input data must be a list of file paths and will hash them as cache keys.

        The filenames of the cached files also try to contain the hash of the transforms. In this
        fashion, `PersistentDataset` should be robust to changes in transforms. This, however, is
        not guaranteed, so caution should be used when modifying transforms to avoid unexpected
        errors. If in doubt, it is advisable to clear the cache directory.

    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        cache_dir: Optional[Union[Path, str]],
        hash_func: Callable[..., bytes] = pickle_hashing,
        pickle_module: str = "pickle",
        pickle_protocol: int = DEFAULT_PROTOCOL,
        hash_transform: Optional[Callable[..., bytes]] = None,
        reset_ops_id: bool = True,
    ) -> None:
        """
        Args:
            data: input data file paths to load and transform to generate dataset for model.
                `PersistentDataset` expects input data to be a list of serializable
                and hashes them as cache keys using `hash_func`.
            transform: transforms to execute operations on input data.
            cache_dir: If specified, this is the location for persistent storage
                of pre-computed transformed data tensors. The cache_dir is computed once, and
                persists on disk until explicitly removed.  Different runs, programs, experiments
                may share a common cache dir provided that the transforms pre-processing is consistent.
                If `cache_dir` doesn't exist, will automatically create it.
                If `cache_dir` is `None`, there is effectively no caching.
            hash_func: a callable to compute hash from data items to be cached.
                defaults to `monai.data.utils.pickle_hashing`.
            pickle_module: string representing the module used for pickling metadata and objects,
                default to `"pickle"`. due to the pickle limitation in multi-processing of Dataloader,
                we can't use `pickle` as arg directly, so here we use a string name instead.
                if want to use other pickle module at runtime, just register like:
                >>> from monai.data import utils
                >>> utils.SUPPORTED_PICKLE_MOD["test"] = other_pickle
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save,
                and ``monai.data.utils.SUPPORTED_PICKLE_MOD``.
            pickle_protocol: can be specified to override the default protocol, default to `2`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.
            hash_transform: a callable to compute hash from the transform information when caching.
                This may reduce errors due to transforms changing during experiments. Default to None (no hash).
                Other options are `pickle_hashing` and `json_hashing` functions from `monai.data.utils`.
            reset_ops_id: whether to set `TraceKeys.ID` to ``Tracekys.NONE``, defaults to ``True``.
                When this is enabled, the traced transform instance IDs will be removed from the cached MetaTensors.
                This is useful for skipping the transform instance checks when inverting applied operations
                using the cached content and with re-created transform instances.

        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hash_func = hash_func
        self.pickle_module = pickle_module
        self.pickle_protocol = pickle_protocol
        if self.cache_dir is not None:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            if not self.cache_dir.is_dir():
                raise ValueError("cache_dir must be a directory.")
        self.transform_hash = ""
        if hash_transform is not None:
            self.set_transform_hash(hash_transform)
        self.reset_ops_id = reset_ops_id

    def set_transform_hash(self, hash_xform_func):
        """Get hashable transforms, and then hash them. Hashable transforms
        are deterministic transforms that inherit from `Transform`. We stop
        at the first non-deterministic transform, or first that does not
        inherit from MONAI's `Transform` class."""
        hashable_transforms = []
        for _tr in self.transform.flatten().transforms:
            if isinstance(_tr, Randomizable) or not isinstance(_tr, Transform):
                break
            hashable_transforms.append(_tr)
        # Try to hash. Fall back to a hash of their names
        try:
            self.transform_hash = hash_xform_func(hashable_transforms)
        except TypeError as te:
            if "is not JSON serializable" not in str(te):
                raise te
            names = "".join(tr.__class__.__name__ for tr in hashable_transforms)
            self.transform_hash = hash_xform_func(names)
        self.transform_hash = self.transform_hash.decode("utf-8")

    def set_data(self, data: Sequence):
        """
        Set the input data and delete all the out-dated cache content.

        """
        self.data = data
        if self.cache_dir is not None and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _pre_transform(self, item_transformed):
        """
        Process the data from original state up to the first random element.

        Args:
            item_transformed: The data to be transformed

        Returns:
            the transformed element up to the first identified
            random transform object

        """
        for _transform in self.transform.transforms:
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            # this is to be consistent with CacheDataset even though it's not in a multi-thread situation.
            _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            item_transformed = apply_transform(_xform, item_transformed)
        if self.reset_ops_id:
            reset_ops_id(item_transformed)
        return item_transformed

    def _post_transform(self, item_transformed):
        """
        Process the data from before the first random transform to the final state ready for evaluation.

        Args:
            item_transformed: The data to be transformed (already processed up to the first random transform)

        Returns:
            the transformed element through the random transforms

        """
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        start_post_randomize_run = False
        for _transform in self.transform.transforms:
            if (
                start_post_randomize_run
                or isinstance(_transform, Randomizable)
                or not isinstance(_transform, Transform)
            ):
                start_post_randomize_run = True
                item_transformed = apply_transform(_transform, item_transformed)
        return item_transformed

    def _cachecheck(self, item_transformed):
        """
        A function to cache the expensive input data transform operations
        so that huge data sets (larger than computer memory) can be processed
        on the fly as needed, and intermediate results written to disk for
        future use.

        Args:
            item_transformed: The current data element to be mutated into transformed representation

        Returns:
            The transformed data_element, either from cache, or explicitly computing it.

        Warning:
            The current implementation does not encode transform information as part of the
            hashing mechanism used for generating cache names when `hash_transform` is None.
            If the transforms applied are changed in any way, the objects in the cache dir will be invalid.

        """
        hashfile = None
        if self.cache_dir is not None:
            data_item_md5 = self.hash_func(item_transformed).decode("utf-8")
            data_item_md5 += self.transform_hash
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            try:
                return torch.load(hashfile)
            except PermissionError as e:
                if sys.platform != "win32":
                    raise e

        _item_transformed = self._pre_transform(deepcopy(item_transformed))  # keep the original hashed
        if hashfile is None:
            return _item_transformed
        try:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(
                    obj=_item_transformed,
                    f=temp_hash_file,
                    pickle_module=look_up_option(self.pickle_module, SUPPORTED_PICKLE_MOD),
                    pickle_protocol=self.pickle_protocol,
                )
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(temp_hash_file, hashfile)
                    except FileExistsError:
                        pass
        except PermissionError:  # project-monai/monai issue #3613
            pass
        return _item_transformed

    def _transform(self, index: int):
        pre_random_item = self._cachecheck(self.data[index])
        return self._post_transform(pre_random_item)


class CacheNTransDataset(PersistentDataset):
    """
    Extension of `PersistentDataset`, tt can also cache the result of first N transforms, no matter it's random or not.

    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        cache_n_trans: int,
        cache_dir: Optional[Union[Path, str]],
        hash_func: Callable[..., bytes] = pickle_hashing,
        pickle_module: str = "pickle",
        pickle_protocol: int = DEFAULT_PROTOCOL,
        hash_transform: Optional[Callable[..., bytes]] = None,
        reset_ops_id: bool = True,
    ) -> None:
        """
        Args:
            data: input data file paths to load and transform to generate dataset for model.
                `PersistentDataset` expects input data to be a list of serializable
                and hashes them as cache keys using `hash_func`.
            transform: transforms to execute operations on input data.
            cache_n_trans: cache the result of first N transforms.
            cache_dir: If specified, this is the location for persistent storage
                of pre-computed transformed data tensors. The cache_dir is computed once, and
                persists on disk until explicitly removed.  Different runs, programs, experiments
                may share a common cache dir provided that the transforms pre-processing is consistent.
                If `cache_dir` doesn't exist, will automatically create it.
                If `cache_dir` is `None`, there is effectively no caching.
            hash_func: a callable to compute hash from data items to be cached.
                defaults to `monai.data.utils.pickle_hashing`.
            pickle_module: string representing the module used for pickling metadata and objects,
                default to `"pickle"`. due to the pickle limitation in multi-processing of Dataloader,
                we can't use `pickle` as arg directly, so here we use a string name instead.
                if want to use other pickle module at runtime, just register like:
                >>> from monai.data import utils
                >>> utils.SUPPORTED_PICKLE_MOD["test"] = other_pickle
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save,
                and ``monai.data.utils.SUPPORTED_PICKLE_MOD``.
            pickle_protocol: can be specified to override the default protocol, default to `2`.
                this arg is used by `torch.save`, for more details, please check:
                https://pytorch.org/docs/stable/generated/torch.save.html#torch.save.
            hash_transform: a callable to compute hash from the transform information when caching.
                This may reduce errors due to transforms changing during experiments. Default to None (no hash).
                Other options are `pickle_hashing` and `json_hashing` functions from `monai.data.utils`.
            reset_ops_id: whether to set `TraceKeys.ID` to ``Tracekys.NONE``, defaults to ``True``.
                When this is enabled, the traced transform instance IDs will be removed from the cached MetaTensors.
                This is useful for skipping the transform instance checks when inverting applied operations
                using the cached content and with re-created transform instances.

        """
        super().__init__(
            data=data,
            transform=transform,
            cache_dir=cache_dir,
            hash_func=hash_func,
            pickle_module=pickle_module,
            pickle_protocol=pickle_protocol,
            hash_transform=hash_transform,
            reset_ops_id=reset_ops_id,
        )
        self.cache_n_trans = cache_n_trans

    def _pre_transform(self, item_transformed):
        """
        Process the data from original state up to the N element.

        Args:
            item_transformed: The data to be transformed

        Returns:
            the transformed element up to the N transform object
        """
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for i, _transform in enumerate(self.transform.transforms):
            if i == self.cache_n_trans:
                break
            _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            item_transformed = apply_transform(_xform, item_transformed)
        reset_ops_id(item_transformed)
        return item_transformed

    def _post_transform(self, item_transformed):
        """
        Process the data from before the N + 1 transform to the final state ready for evaluation.

        Args:
            item_transformed: The data to be transformed (already processed up to the first N transform)

        Returns:
            the final transformed result
        """
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for i, _transform in enumerate(self.transform.transforms):
            if i >= self.cache_n_trans:
                item_transformed = apply_transform(_transform, item_transformed)
        return item_transformed


class LMDBDataset(PersistentDataset):
    """
    Extension of `PersistentDataset` using LMDB as the backend.

    See Also:
        :py:class:`monai.data.PersistentDataset`

    Examples:

        >>> items = [{"data": i} for i in range(5)]
        # [{'data': 0}, {'data': 1}, {'data': 2}, {'data': 3}, {'data': 4}]
        >>> lmdb_ds = monai.data.LMDBDataset(items, transform=monai.transforms.SimulateDelayd("data", delay_time=1))
        >>> print(list(lmdb_ds))  # using the cached results

    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        cache_dir: Union[Path, str] = "cache",
        hash_func: Callable[..., bytes] = pickle_hashing,
        db_name: str = "monai_cache",
        progress: bool = True,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
        hash_transform: Optional[Callable[..., bytes]] = None,
        reset_ops_id: bool = True,
        lmdb_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Args:
            data: input data file paths to load and transform to generate dataset for model.
                `LMDBDataset` expects input data to be a list of serializable
                and hashes them as cache keys using `hash_func`.
            transform: transforms to execute operations on input data.
            cache_dir: if specified, this is the location for persistent storage
                of pre-computed transformed data tensors. The cache_dir is computed once, and
                persists on disk until explicitly removed.  Different runs, programs, experiments
                may share a common cache dir provided that the transforms pre-processing is consistent.
                If the cache_dir doesn't exist, will automatically create it. Defaults to "./cache".
            hash_func: a callable to compute hash from data items to be cached.
                defaults to `monai.data.utils.pickle_hashing`.
            db_name: lmdb database file name. Defaults to "monai_cache".
            progress: whether to display a progress bar.
            pickle_protocol: pickle protocol version. Defaults to pickle.HIGHEST_PROTOCOL.
                https://docs.python.org/3/library/pickle.html#pickle-protocols
            hash_transform: a callable to compute hash from the transform information when caching.
                This may reduce errors due to transforms changing during experiments. Default to None (no hash).
                Other options are `pickle_hashing` and `json_hashing` functions from `monai.data.utils`.
            reset_ops_id: whether to set `TraceKeys.ID` to ``Tracekys.NONE``, defaults to ``True``.
                When this is enabled, the traced transform instance IDs will be removed from the cached MetaTensors.
                This is useful for skipping the transform instance checks when inverting applied operations
                using the cached content and with re-created transform instances.
            lmdb_kwargs: additional keyword arguments to the lmdb environment.
                for more details please visit: https://lmdb.readthedocs.io/en/release/#environment-class
        """
        super().__init__(
            data=data,
            transform=transform,
            cache_dir=cache_dir,
            hash_func=hash_func,
            pickle_protocol=pickle_protocol,
            hash_transform=hash_transform,
            reset_ops_id=reset_ops_id,
        )
        self.progress = progress
        if not self.cache_dir:
            raise ValueError("cache_dir must be specified.")
        self.db_file = self.cache_dir / f"{db_name}.lmdb"
        self.lmdb_kwargs = lmdb_kwargs or {}
        if not self.lmdb_kwargs.get("map_size", 0):
            self.lmdb_kwargs["map_size"] = 1024**4  # default map_size
        # lmdb is single-writer multi-reader by default
        # the cache is created without multi-threading
        self._read_env = None
        # this runs on the primary thread/process
        self._fill_cache_start_reader(show_progress=self.progress)
        print(f"Accessing lmdb file: {self.db_file.absolute()}.")

    def set_data(self, data: Sequence):
        """
        Set the input data and delete all the out-dated cache content.

        """
        super().set_data(data=data)
        self._read_env = self._fill_cache_start_reader(show_progress=self.progress)

    def _fill_cache_start_reader(self, show_progress=True):
        """
        Check the LMDB cache and write the cache if needed. py-lmdb doesn't have a good support for concurrent write.
        This method can be used with multiple processes, but it may have a negative impact on the performance.

        Args:
            show_progress: whether to show the progress bar if possible.
        """
        # create cache
        self.lmdb_kwargs["readonly"] = False
        env = lmdb.open(path=f"{self.db_file}", subdir=False, **self.lmdb_kwargs)
        if show_progress and not has_tqdm:
            warnings.warn("LMDBDataset: tqdm is not installed. not displaying the caching progress.")
        with env.begin(write=False) as search_txn:
            for item in tqdm(self.data) if has_tqdm and show_progress else self.data:
                key = self.hash_func(item)
                done, retry, val = False, 5, None
                while not done and retry > 0:
                    try:
                        with search_txn.cursor() as cursor:
                            done = cursor.set_key(key)
                        if done:
                            continue
                        if val is None:
                            val = self._pre_transform(deepcopy(item))  # keep the original hashed
                            val = pickle.dumps(val, protocol=self.pickle_protocol)
                        with env.begin(write=True) as txn:
                            txn.put(key, val)
                        done = True
                    except lmdb.MapFullError:
                        done, retry = False, retry - 1
                        size = env.info()["map_size"]
                        new_size = size * 2
                        warnings.warn(
                            f"Resizing the cache database from {int(size) >> 20}MB" f" to {int(new_size) >> 20}MB."
                        )
                        env.set_mapsize(new_size)
                    except lmdb.MapResizedError:
                        # the mapsize is increased by another process
                        # set_mapsize with a size of 0 to adopt the new size
                        env.set_mapsize(0)
                if not done:  # still has the map full error
                    size = env.info()["map_size"]
                    env.close()
                    raise ValueError(f"LMDB map size reached, increase size above current size of {size}.")
        size = env.info()["map_size"]
        env.close()
        # read-only database env
        self.lmdb_kwargs["readonly"] = True
        self.lmdb_kwargs["map_size"] = size
        if self.lmdb_kwargs.get("lock", None) is None:
            self.lmdb_kwargs["lock"] = False
        if self.lmdb_kwargs.get("readahead", None) is None:
            self.lmdb_kwargs["readahead"] = False
        return lmdb.open(path=f"{self.db_file}", subdir=False, **self.lmdb_kwargs)

    def _cachecheck(self, item_transformed):
        """
        if the item is not found in the lmdb file, resolves to the persistent cache default behaviour.

        """
        if self._read_env is None:
            # this runs on multiple processes, each one should have its own env.
            self._read_env = self._fill_cache_start_reader(show_progress=False)
        with self._read_env.begin(write=False) as txn:
            data = txn.get(self.hash_func(item_transformed))
        if data is None:
            warnings.warn("LMDBDataset: cache key not found, running fallback caching.")
            return super()._cachecheck(item_transformed)
        try:
            return pickle.loads(data)
        except Exception as err:
            raise RuntimeError("Invalid cache value, corrupted lmdb file?") from err

    def info(self):
        """
        Returns: dataset info dictionary.

        """
        if self._read_env is None:
            self._read_env = self._fill_cache_start_reader()
        out = dict(self._read_env.info())
        out["size"] = len(self.data)
        out["filename"] = f"{self.db_file.absolute()}"
        return out


class CacheDataset(Dataset):
    """
    Dataset with cache mechanism that can load data and cache deterministic transforms' result during training.

    By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline.
    If the requested data is not in the cache, all transforms will run normally
    (see also :py:class:`monai.data.dataset.Dataset`).

    Users can set the cache rate or number of items to cache.
    It is recommended to experiment with different `cache_num` or `cache_rate` to identify the best training speed.

    The transforms which are supposed to be cached must implement the `monai.transforms.Transform`
    interface and should not be `Randomizable`. This dataset will cache the outcomes before the first
    `Randomizable` `Transform` within a `Compose` instance.
    So to improve the caching efficiency, please always put as many as possible non-random transforms
    before the randomized ones when composing the chain of transforms.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, if the transform is a `Compose` of::

        transforms = Compose([
            LoadImaged(),
            EnsureChannelFirstd(),
            Spacingd(),
            Orientationd(),
            ScaleIntensityRanged(),
            RandCropByPosNegLabeld(),
            ToTensord()
        ])

    when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
    this dataset will cache the results up to ``ScaleIntensityRanged``, as
    all non-random transforms `LoadImaged`, `EnsureChannelFirstd`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
    can be cached. During training, the dataset will load the cached results and run
    ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
    and the outcome not cached.

    During training call `set_data()` to update input data and recompute cache content, note that it requires
    `persistent_workers=False` in the PyTorch DataLoader.

    Note:
        `CacheDataset` executes non-random transforms and prepares cache content in the main process before
        the first epoch, then all the subprocesses of DataLoader will read the same cache content in the main process
        during training. it may take a long time to prepare cache content according to the size of expected cache data.
        So to debug or verify the program before real training, users can set `cache_rate=0.0` or `cache_num=0` to
        temporarily skip caching.

    """

    def __init__(
        self,
        data: Sequence,
        transform: Optional[Union[Sequence[Callable], Callable]] = None,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: Optional[int] = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        hash_as_key: bool = False,
        hash_func: Callable[..., bytes] = pickle_hashing,
        runtime_cache: bool = False,
    ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: transforms to execute operations on input data.
            cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers: the number of worker threads to use.
                If num_workers is None then the number returned by os.cpu_count() is used.
                If a value less than 1 is speficied, 1 will be used instead.
            progress: whether to display a progress bar.
            copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
                default to `True`. if the random transforms don't modify the cached content
                (for example, randomly crop from the cached image and deepcopy the crop region)
                or if every cache item is only used once in a `multi-processing` environment,
                may set `copy=False` for better performance.
            as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
                it may help improve the performance of following logic.
            hash_as_key: whether to compute hash value of input data as the key to save cache,
                if key exists, avoid saving duplicated content. it can help save memory when
                the dataset has duplicated items or augmented dataset.
            hash_func: if `hash_as_key`, a callable to compute hash from data items to be cached.
                defaults to `monai.data.utils.pickle_hashing`.

        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)
        self.set_num = cache_num  # tracking the user-provided `cache_num` option
        self.set_rate = cache_rate  # tracking the user-provided `cache_rate` option
        self.progress = progress
        self.copy_cache = copy_cache
        self.as_contiguous = as_contiguous
        self.hash_as_key = hash_as_key
        self.hash_func = hash_func
        self.num_workers = num_workers
        if self.num_workers is not None:
            self.num_workers = max(int(self.num_workers), 1)
        self.cache_num = 0
        self._cache: Union[List, Dict] = []
        self.set_data(data)

    def set_data(self, data: Sequence):
        """
        Set the input data and run deterministic transforms to generate cache content.

        Note: should call this func after an entire epoch and must set `persistent_workers=False`
        in PyTorch DataLoader, because it needs to create new worker processes based on new
        generated cache content.

        """

        def _compute_cache():
            self.cache_num = min(int(self.set_num), int(len(self.data) * self.set_rate), len(self.data))
            return self._fill_cache()

        if self.hash_as_key:
            # only compute cache for the unique items of dataset
            mapping = {self.hash_func(v): v for v in data}
            self.data = list(mapping.values())
            cache_ = _compute_cache()
            self._cache = Manager().dict(zip(list(mapping)[: self.cache_num], cache_))
            self.data = data
        else:
            self.data = data
            self._cache = Manager().list(_compute_cache())

    def set_multiprocessing_cache(self, mp_cache: bool = True):
        self.mp_cache = mp_cache

    def _fill_cache(self) -> List:
        if self.cache_num <= 0:
            return []
        if self.progress and not has_tqdm:
            warnings.warn("tqdm is not installed, will not show the caching progress bar.")
        with ThreadPool(self.num_workers) as p:
            if self.progress and has_tqdm:
                return list(
                    tqdm(
                        p.imap(self._load_cache_item, range(self.cache_num)),
                        total=self.cache_num,
                        desc="Loading dataset",
                    )
                )
            return list(p.imap(self._load_cache_item, range(self.cache_num)))

    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        item = self.data[idx]
        for _transform in self.transform.transforms:  # type:ignore
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            item = apply_transform(_xform, item)
        if self.as_contiguous:
            item = convert_to_contiguous(item, memory_format=torch.contiguous_format)
        return item

    def _transform(self, index: int):
        index_: Any = index
        if self.hash_as_key:
            key = self.hash_func(self.data[index])
            if key in self._cache:
                # if existing in cache, get the index
                index_ = key  # if using hash as cache keys, set the key

        if isinstance(index_, int) and index_ % len(self) >= self.cache_num:  # support negative index
            # no cache for this index, execute all the transforms directly
            return super()._transform(index_)
        # load data from cache and execute from the first random transform
        start_run = False
        if self._cache is None:
            self._cache = self._fill_cache()
        data = self._cache[index_]
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for _transform in self.transform.transforms:
            if start_run or isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                # only need to deep copy data on first non-deterministic transform
                if not start_run:
                    start_run = True
                    if self.copy_cache:
                        data = deepcopy(data)
                data = apply_transform(_transform, data)
        return data


class SmartCacheDataset(Randomizable, CacheDataset):
    """
    Re-implementation of the SmartCache mechanism in NVIDIA Clara-train SDK.
    At any time, the cache pool only keeps a subset of the whole dataset. In each epoch, only the items
    in the cache are used for training. This ensures that data needed for training is readily available,
    keeping GPU resources busy. Note that cached items may still have to go through a non-deterministic
    transform sequence before being fed to GPU. At the same time, another thread is preparing replacement
    items by applying the transform sequence to items not in cache. Once one epoch is completed, Smart
    Cache replaces the same number of items with replacement items.
    Smart Cache uses a simple `running window` algorithm to determine the cache content and replacement items.
    Let N be the configured number of objects in cache; and R be the number of replacement objects (R = ceil(N * r),
    where r is the configured replace rate).
    For more details, please refer to:
    https://docs.nvidia.com/clara/tlt-mi/clara-train-sdk-v3.0/nvmidl/additional_features/smart_cache.html#smart-cache
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, if we have 5 images: `[image1, image2, image3, image4, image5]`, and `cache_num=4`, `replace_rate=0.25`.
    so the actual training images cached and replaced for every epoch are as below::

        epoch 1: [image1, image2, image3, image4]
        epoch 2: [image2, image3, image4, image5]
        epoch 3: [image3, image4, image5, image1]
        epoch 3: [image4, image5, image1, image2]
        epoch N: [image[N % 5] ...]

    The usage of `SmartCacheDataset` contains 4 steps:

        1. Initialize `SmartCacheDataset` object and cache for the first epoch.
        2. Call `start()` to run replacement thread in background.
        3. Call `update_cache()` before every epoch to replace training items.
        4. Call `shutdown()` when training ends.

    During training call `set_data()` to update input data and recompute cache content, note to call
    `shutdown()` to stop first, then update data and call `start()` to restart.

    Note:
        This replacement will not work for below cases:
        1. Set the `multiprocessing_context` of DataLoader to `spawn`.
        2. Launch distributed data parallel with `torch.multiprocessing.spawn`.
        3. Run on windows(the default multiprocessing method is `spawn`) with `num_workers` greater than 0.
        4. Set the `persistent_workers` of DataLoader to `True` with `num_workers` greater than 0.

        If using MONAI workflows, please add `SmartCacheHandler` to the handler list of trainer,
        otherwise, please make sure to call `start()`, `update_cache()`, `shutdown()` during training.

    Args:
        data: input data to load and transform to generate dataset for model.
        transform: transforms to execute operations on input data.
        replace_rate: percentage of the cached items to be replaced in every epoch (default to 0.1).
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_init_workers: the number of worker threads to initialize the cache for first epoch.
            If num_init_workers is None then the number returned by os.cpu_count() is used.
            If a value less than 1 is speficied, 1 will be used instead.
        num_replace_workers: the number of worker threads to prepare the replacement cache for every epoch.
            If num_replace_workers is None then the number returned by os.cpu_count() is used.
            If a value less than 1 is speficied, 1 will be used instead.
        progress: whether to display a progress bar when caching for the first epoch.
        shuffle: whether to shuffle the whole data list before preparing the cache content for first epoch.
            it will not modify the original input data sequence in-place.
        seed: random seed if shuffle is `True`, default to `0`.
        copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
            default to `True`. if the random transforms don't modify the cache content
            or every cache item is only used once in a `multi-processing` environment,
            may set `copy=False` for better performance.
        as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
            it may help improve the performance of following logic.

    """

    def __init__(
        self,
        data: Sequence,
        transform: Optional[Union[Sequence[Callable], Callable]] = None,
        replace_rate: float = 0.1,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_init_workers: Optional[int] = 1,
        num_replace_workers: Optional[int] = 1,
        progress: bool = True,
        shuffle: bool = True,
        seed: int = 0,
        copy_cache: bool = True,
        as_contiguous: bool = True,
    ) -> None:
        if shuffle:
            self.set_random_state(seed=seed)
        self.shuffle = shuffle

        self._start_pos: int = 0
        self._update_lock: threading.Lock = threading.Lock()
        self._round: int = 1
        self._replace_done: bool = False
        self._replace_mgr: Optional[threading.Thread] = None

        super().__init__(data, transform, cache_num, cache_rate, num_init_workers, progress, copy_cache, as_contiguous)
        if self._cache is None:
            self._cache = self._fill_cache()
        if self.cache_num >= len(data):
            warnings.warn(
                "cache_num is greater or equal than dataset length, fall back to regular monai.data.CacheDataset."
            )
        if replace_rate <= 0:
            raise ValueError("replace_rate must be greater than 0, otherwise, please use monai.data.CacheDataset.")

        self.num_replace_workers: Optional[int] = num_replace_workers
        if self.num_replace_workers is not None:
            self.num_replace_workers = max(int(self.num_replace_workers), 1)

        self._total_num: int = len(data)
        self._replace_num: int = min(math.ceil(self.cache_num * replace_rate), len(data) - self.cache_num)
        self._replacements: List[Any] = [None for _ in range(self._replace_num)]
        self._replace_data_idx: List[int] = list(range(self._replace_num))
        self._compute_data_idx()

    def set_data(self, data: Sequence):
        """
        Set the input data and run deterministic transforms to generate cache content.

        Note: should call `shutdown()` before calling this func.

        """
        if self.is_started():
            warnings.warn("SmartCacheDataset is not shutdown yet, shutdown it directly.")
            self.shutdown()

        if self.shuffle:
            data = copy(data)
            self.randomize(data)
        super().set_data(data)

    def randomize(self, data: Sequence) -> None:
        try:
            self.R.shuffle(data)
        except TypeError as e:
            warnings.warn(f"input data can't be shuffled in SmartCacheDataset with numpy.random.shuffle(): {e}.")

    def _compute_data_idx(self):
        """
        Update the replacement data position in the total data.

        """
        for i in range(self._replace_num):
            pos: int = self._start_pos + self.cache_num + i
            if pos >= self._total_num:
                pos -= self._total_num
            self._replace_data_idx[i] = pos

    def is_started(self):
        """
        Check whether the replacement thread is already started.

        """
        if self._replace_mgr is None:
            return False
        return self._replace_mgr.is_alive()

    def start(self):
        """
        Start the background thread to replace training items for every epoch.

        """
        if self._replace_mgr is None or not self.is_started():
            self._restart()

    def _restart(self):
        """
        Restart background thread if killed for some reason.

        """
        self._round = 1
        self._replace_mgr = threading.Thread(target=self.manage_replacement, daemon=True)
        self._replace_mgr.start()

    def _try_update_cache(self):
        """
        Update the cache items with new replacement for current epoch.

        """
        with self._update_lock:
            if not self._replace_done:
                return False

            del self._cache[: self._replace_num]
            self._cache.extend(self._replacements)

            self._start_pos += self._replace_num
            if self._start_pos >= self._total_num:
                self._start_pos -= self._total_num

            self._compute_data_idx()

            # ready for next round
            self._round += 1
            self._replace_done = False
            return True

    def update_cache(self):
        """
        Update cache items for current epoch, need to call this function before every epoch.
        If the cache has been shutdown before, need to restart the `_replace_mgr` thread.

        """
        if not self._replace_mgr.is_alive():
            self._restart()

        # make sure update is done
        while not self._try_update_cache():
            time.sleep(0.01)

    def _try_shutdown(self):
        """
        Wait for thread lock to shut down the background thread.

        """
        with self._update_lock:
            if self._replace_done:
                self._round = 0
                self._start_pos = 0
                self._compute_data_idx()
                self._replace_done = False
                return True
            return False

    def shutdown(self):
        """
        Shut down the background thread for replacement.

        """
        if not self.is_started():
            return

        # wait until replace mgr is done the current round
        while not self._try_shutdown():
            time.sleep(0.01)
        self._replace_mgr.join()

    def _replace_cache_thread(self, index: int):
        """
        Execute deterministic transforms on the new data for replacement.

        """
        pos: int = self._replace_data_idx[index]
        self._replacements[index] = self._load_cache_item(pos)

    def _compute_replacements(self):
        """
        Compute expected items for the replacement of next epoch, execute deterministic transforms.
        It can support multi-threads to accelerate the computation progress.

        """
        with ThreadPool(self.num_replace_workers) as p:
            p.map(self._replace_cache_thread, list(range(self._replace_num)))

        self._replace_done = True

    def _try_manage_replacement(self, check_round):
        """
        Wait thread lock and replace training items in the background thread.

        """
        with self._update_lock:
            if self._round <= 0:
                # shutdown replacement
                self._replace_done = True
                return True, -1

            if self._round != check_round:
                self._compute_replacements()
            return False, self._round

    def manage_replacement(self):
        """
        Background thread for replacement.

        """
        check_round: int = -1
        done = False
        while not done:
            done, check_round = self._try_manage_replacement(check_round)
            time.sleep(0.01)

    def __len__(self):
        """
        The dataset length is given by cache_num instead of len(data).

        """
        return self.cache_num


class ZipDataset(Dataset):
    """
    Zip several PyTorch datasets and output data(with the same index) together in a tuple.
    If the output of single dataset is already a tuple, flatten it and extend to the result.
    For example: if datasetA returns (img, imgmeta), datasetB returns (seg, segmeta),
    finally return (img, imgmeta, seg, segmeta).
    And if the datasets don't have same length, use the minimum length of them as the length
    of ZipDataset.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    Examples::

        >>> zip_data = ZipDataset([[1, 2, 3], [4, 5]])
        >>> print(len(zip_data))
        2
        >>> for item in zip_data:
        >>>    print(item)
        [1, 4]
        [2, 5]

    """

    def __init__(self, datasets: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            datasets: list of datasets to zip together.
            transform: a callable data transform operates on the zipped item from `datasets`.
        """
        super().__init__(list(datasets), transform=transform)

    def __len__(self) -> int:
        return min(len(dataset) for dataset in self.data)

    def _transform(self, index: int):
        def to_list(x):
            return list(x) if isinstance(x, (tuple, list)) else [x]

        data = []
        for dataset in self.data:
            data.extend(to_list(dataset[index]))
        if self.transform is not None:
            data = apply_transform(self.transform, data, map_items=False)  # transform the list data
        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        return tuple(data)


class ArrayDataset(Randomizable, _TorchDataset):
    """
    Dataset for segmentation and classification tasks based on array format input data and transforms.
    It ensures the same random seeds in the randomized transforms defined for image, segmentation and label.
    The `transform` can be :py:class:`monai.transforms.Compose` or any other callable object.
    For example:
    If train based on Nifti format images without metadata, all transforms can be composed::

        img_transform = Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                RandAdjustContrast()
            ]
        )
        ArrayDataset(img_file_list, img_transform=img_transform)

    If training based on images and the metadata, the array transforms can not be composed
    because several transforms receives multiple parameters or return multiple values. Then Users need
    to define their own callable method to parse metadata from `LoadImage` or set `affine` matrix
    to `Spacing` transform::

        class TestCompose(Compose):
            def __call__(self, input_):
                img, metadata = self.transforms[0](input_)
                img = self.transforms[1](img)
                img, _, _ = self.transforms[2](img, metadata["affine"])
                return self.transforms[3](img), metadata
        img_transform = TestCompose(
            [
                LoadImage(image_only=False),
                EnsureChannelFirst(),
                Spacing(pixdim=(1.5, 1.5, 3.0)),
                RandAdjustContrast()
            ]
        )
        ArrayDataset(img_file_list, img_transform=img_transform)

    Examples::

        >>> ds = ArrayDataset([1, 2, 3, 4], lambda x: x + 0.1)
        >>> print(ds[0])
        1.1

        >>> ds = ArrayDataset(img=[1, 2, 3, 4], seg=[5, 6, 7, 8])
        >>> print(ds[0])
        [1, 5]

    """

    def __init__(
        self,
        img: Sequence,
        img_transform: Optional[Callable] = None,
        seg: Optional[Sequence] = None,
        seg_transform: Optional[Callable] = None,
        labels: Optional[Sequence] = None,
        label_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the dataset with the filename lists. The transform `img_transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            img: sequence of images.
            img_transform: transform to apply to each element in `img`.
            seg: sequence of segmentations.
            seg_transform: transform to apply to each element in `seg`.
            labels: sequence of labels.
            label_transform: transform to apply to each element in `labels`.

        """
        items = [(img, img_transform), (seg, seg_transform), (labels, label_transform)]
        self.set_random_state(seed=get_seed())
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.dataset)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        return self.dataset[index]


class NPZDictItemDataset(Dataset):
    """
    Represents a dataset from a loaded NPZ file. The members of the file to load are named in the keys of `keys` and
    stored under the keyed name. All loaded arrays must have the same 0-dimension (batch) size. Items are always dicts
    mapping names to an item extracted from the loaded arrays.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    Args:
        npzfile: Path to .npz file or stream containing .npz file data
        keys: Maps keys to load from file to name to store in dataset
        transform: Transform to apply to batch dict
        other_keys: secondary data to load from file and store in dict `other_keys`, not returned by __getitem__
    """

    def __init__(
        self,
        npzfile: Union[str, IO],
        keys: Dict[str, str],
        transform: Optional[Callable[..., Dict[str, Any]]] = None,
        other_keys: Optional[Sequence[str]] = (),
    ):
        self.npzfile: Union[str, IO] = npzfile if isinstance(npzfile, str) else "STREAM"
        self.keys: Dict[str, str] = dict(keys)
        dat = np.load(npzfile)

        self.arrays = {storedk: dat[datak] for datak, storedk in self.keys.items()}
        self.length = self.arrays[first(self.keys.values())].shape[0]

        self.other_keys = {} if other_keys is None else {k: dat[k] for k in other_keys}

        for k, v in self.arrays.items():
            if v.shape[0] != self.length:
                raise ValueError(
                    "All loaded arrays must have the same first dimension "
                    f"size {self.length}, array `{k}` has size {v.shape[0]}"
                )

        super().__init__([], transform)

    def __len__(self):
        return self.length

    def _transform(self, index: int):
        data = {k: v[index] for k, v in self.arrays.items()}

        if not self.transform:
            return data

        result = apply_transform(self.transform, data)

        if isinstance(result, dict) or (isinstance(result, list) and isinstance(result[0], dict)):
            return result
        raise AssertionError("With a dict supplied to apply_transform, should return a dict or a list of dicts.")


class CSVDataset(Dataset):
    """
    Dataset to load data from CSV files and generate a list of dictionaries,
    every dictionary maps to a row of the CSV file, and the keys of dictionary
    map to the column names of the CSV file.

    It can load multiple CSV files and join the tables with additional `kwargs` arg.
    Support to only load specific rows and columns.
    And it can also group several loaded columns to generate a new column, for example,
    set `col_groups={"meta": ["meta_0", "meta_1", "meta_2"]}`, output can be::

        [
            {"image": "./image0.nii", "meta_0": 11, "meta_1": 12, "meta_2": 13, "meta": [11, 12, 13]},
            {"image": "./image1.nii", "meta_0": 21, "meta_1": 22, "meta_2": 23, "meta": [21, 22, 23]},
        ]

    Args:
        src: if provided the filename of CSV file, it can be a str, URL, path object or file-like object to load.
            also support to provide pandas `DataFrame` directly, will skip loading from filename.
            if provided a list of filenames or pandas `DataFrame`, it will join the tables.
        row_indices: indices of the expected rows to load. it should be a list,
            every item can be a int number or a range `[start, end)` for the indices.
            for example: `row_indices=[[0, 100], 200, 201, 202, 300]`. if None,
            load all the rows in the file.
        col_names: names of the expected columns to load. if None, load all the columns.
        col_types: `type` and `default value` to convert the loaded columns, if None, use original data.
            it should be a dictionary, every item maps to an expected column, the `key` is the column
            name and the `value` is None or a dictionary to define the default value and data type.
            the supported keys in dictionary are: ["type", "default"]. for example::

                col_types = {
                    "subject_id": {"type": str},
                    "label": {"type": int, "default": 0},
                    "ehr_0": {"type": float, "default": 0.0},
                    "ehr_1": {"type": float, "default": 0.0},
                    "image": {"type": str, "default": None},
                }

        col_groups: args to group the loaded columns to generate a new column,
            it should be a dictionary, every item maps to a group, the `key` will
            be the new column name, the `value` is the names of columns to combine. for example:
            `col_groups={"ehr": [f"ehr_{i}" for i in range(10)], "meta": ["meta_1", "meta_2"]}`
        transform: transform to apply on the loaded items of a dictionary data.
        kwargs_read_csv: dictionary args to pass to pandas `read_csv` function.
        kwargs: additional arguments for `pandas.merge()` API to join tables.

    .. deprecated:: 0.8.0
        ``filename`` is deprecated, use ``src`` instead.

    """

    @deprecated_arg(name="filename", new_name="src", since="0.8", msg_suffix="please use `src` instead.")
    def __init__(
        self,
        src: Optional[Union[str, Sequence[str]]] = None,  # also can be `DataFrame` or a sequence of `DataFrame`
        row_indices: Optional[Sequence[Union[int, str]]] = None,
        col_names: Optional[Sequence[str]] = None,
        col_types: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        col_groups: Optional[Dict[str, Sequence[str]]] = None,
        transform: Optional[Callable] = None,
        kwargs_read_csv: Optional[Dict] = None,
        **kwargs,
    ):
        srcs = (src,) if not isinstance(src, (tuple, list)) else src
        dfs: List = []
        for i in srcs:
            if isinstance(i, str):
                dfs.append(pd.read_csv(i, **kwargs_read_csv) if kwargs_read_csv else pd.read_csv(i))
            elif isinstance(i, pd.DataFrame):
                dfs.append(i)
            else:
                raise ValueError("`src` must be file path or pandas `DataFrame`.")

        # in case treating deprecated arg `filename` as kwargs, remove it from `kwargs`
        kwargs.pop("filename", None)

        data = convert_tables_to_dicts(
            dfs=dfs, row_indices=row_indices, col_names=col_names, col_types=col_types, col_groups=col_groups, **kwargs
        )
        super().__init__(data=data, transform=transform)
