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


import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset

from monai.data.utils import first, pickle_hashing
from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform
from monai.utils import MAX_SEED, get_seed, min_version, optional_import

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

lmdb, _ = optional_import("lmdb")


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


class PersistentDataset(Dataset):
    """
    Persistent storage of pre-computed values to efficiently manage larger than memory dictionary format data,
    it can operate transforms for specific fields.  Results from the non-random transform components are computed
    when first used, and stored in the `cache_dir` for rapid retrieval on subsequent uses.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
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

    Note:
        The input data must be a list of file paths and will hash them as cache keys.

        When loading persistent cache content, it can't guarantee the cached data matches current
        transform chain, so please make sure to use exactly the same non-random transforms and the
        args as the cache content, otherwise, it may cause unexpected errors.

    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        cache_dir: Optional[Union[Path, str]],
        hash_func: Callable[..., bytes] = pickle_hashing,
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

        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hash_func = hash_func
        if self.cache_dir is not None:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True)
            if not self.cache_dir.is_dir():
                raise ValueError("cache_dir must be a directory.")

    def _pre_transform(self, item_transformed):
        """
        Process the data from original state up to the first random element.

        Args:
            item_transformed: The data to be transformed

        Returns:
            the transformed element up to the first identified
            random transform object

        """
        for _transform in self.transform.transforms:  # type:ignore
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            # this is to be consistent with CacheDataset even though it's not in a multi-thread situation.
            _xform = deepcopy(_transform) if isinstance(_transform, ThreadUnsafe) else _transform
            item_transformed = apply_transform(_xform, item_transformed)
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
            hashing mechanism used for generating cache names.  If the transforms applied are
            changed in any way, the objects in the cache dir will be invalid.  The hash for the
            cache is ONLY dependant on the input filename paths.

        """
        hashfile = None
        if self.cache_dir is not None:
            data_item_md5 = self.hash_func(item_transformed).decode("utf-8")
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        if hashfile is not None and hashfile.is_file():  # cache hit
            return torch.load(hashfile)

        _item_transformed = self._pre_transform(deepcopy(item_transformed))  # keep the original hashed
        if hashfile is not None:
            # NOTE: Writing to a temporary directory and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_hash_file = Path(tmpdirname) / hashfile.name
                torch.save(_item_transformed, temp_hash_file)
                if temp_hash_file.is_file() and not hashfile.is_file():
                    # On Unix, if target exists and is a file, it will be replaced silently if the user has permission.
                    # for more details: https://docs.python.org/3/library/shutil.html#shutil.move.
                    try:
                        shutil.move(temp_hash_file, hashfile)
                    except FileExistsError:
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

        """
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, hash_func=hash_func)
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
            lmdb_kwargs: additional keyword arguments to the lmdb environment.
                for more details please visit: https://lmdb.readthedocs.io/en/release/#environment-class
        """
        super().__init__(data=data, transform=transform, cache_dir=cache_dir, hash_func=hash_func)
        self.progress = progress
        if not self.cache_dir:
            raise ValueError("cache_dir must be specified.")
        self.db_file = self.cache_dir / f"{db_name}.lmdb"
        self.pickle_protocol = pickle_protocol
        self.lmdb_kwargs = lmdb_kwargs or {}
        if not self.lmdb_kwargs.get("map_size", 0):
            self.lmdb_kwargs["map_size"] = 1024 ** 4  # default map_size
        self._read_env = None
        print(f"Accessing lmdb file: {self.db_file.absolute()}.")

    def _fill_cache_start_reader(self):
        # create cache
        self.lmdb_kwargs["readonly"] = False
        env = lmdb.open(path=f"{self.db_file}", subdir=False, **self.lmdb_kwargs)
        if self.progress and not has_tqdm:
            warnings.warn("LMDBDataset: tqdm is not installed. not displaying the caching progress.")
        for item in tqdm(self.data) if has_tqdm and self.progress else self.data:
            key = self.hash_func(item)
            done, retry, val = False, 5, None
            while not done and retry > 0:
                try:
                    with env.begin(write=True) as txn:
                        with txn.cursor() as cursor:
                            done = cursor.set_key(key)
                            if done:
                                continue
                        if val is None:
                            val = self._pre_transform(deepcopy(item))  # keep the original hashed
                            val = pickle.dumps(val, protocol=self.pickle_protocol)
                        txn.put(key, val)
                    done = True
                except lmdb.MapFullError:
                    done, retry = False, retry - 1
                    size = env.info()["map_size"]
                    new_size = size * 2
                    warnings.warn(f"Resizing the cache database from {int(size) >> 20}MB to {int(new_size) >> 20}MB.")
                    env.set_mapsize(new_size)
                except lmdb.MapResizedError:
                    # the mapsize is increased by another process
                    # set_mapsize with a size of 0 to adopt the new size,
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
            self._read_env = self._fill_cache_start_reader()
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

    To improve the caching efficiency, please always put as many as possible non-random transforms
    before the randomized ones when composing the chain of transforms.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, if the transform is a `Compose` of::

        transforms = Compose([
            LoadImaged(),
            AddChanneld(),
            Spacingd(),
            Orientationd(),
            ScaleIntensityRanged(),
            RandCropByPosNegLabeld(),
            ToTensord()
        ])

    when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
    this dataset will cache the results up to ``ScaleIntensityRanged``, as
    all non-random transforms `LoadImaged`, `AddChanneld`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
    can be cached. During training, the dataset will load the cached results and run
    ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
    and the outcome not cached.

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
        transform: Union[Sequence[Callable], Callable],
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: Optional[int] = None,
        progress: bool = True,
    ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: transforms to execute operations on input data.
            cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_workers: the number of worker processes to use.
                If num_workers is None then the number returned by os.cpu_count() is used.
            progress: whether to display a progress bar.
        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)
        self.progress = progress
        self.cache_num = min(int(cache_num), int(len(data) * cache_rate), len(data))
        self.num_workers = num_workers
        if self.num_workers is not None:
            self.num_workers = max(int(self.num_workers), 1)
        self._cache: List = self._fill_cache()

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
        return item

    def _transform(self, index: int):
        if index % len(self) >= self.cache_num:  # support negative index
            # no cache for this index, execute all the transforms directly
            return super()._transform(index)
        # load data from cache and execute from the first random transform
        start_run = False
        if self._cache is None:
            self._cache = self._fill_cache()
        data = self._cache[index]
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for _transform in self.transform.transforms:
            if start_run or isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                # only need to deep copy data on first non-deterministic transform
                if not start_run:
                    start_run = True
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

    Note:
        This replacement will not work for below cases:
        1. Set the `multiprocessing_context` of DataLoader to `spawn`.
        2. Run on windows(the default multiprocessing method is `spawn`) with `num_workers` greater than 0.
        3. Set the `persistent_workers` of DataLoader to `True` with `num_workers` greater than 0.

        If using MONAI workflows, please add `SmartCacheHandler` to the handler list of trainer,
        otherwise, please make sure to call `start()`, `update_cache()`, `shutdown()` during training.

    Args:
        data: input data to load and transform to generate dataset for model.
        transform: transforms to execute operations on input data.
        replace_rate: percentage of the cached items to be replaced in every epoch.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_init_workers: the number of worker threads to initialize the cache for first epoch.
            If num_init_workers is None then the number returned by os.cpu_count() is used.
        num_replace_workers: the number of worker threads to prepare the replacement cache for every epoch.
            If num_replace_workers is None then the number returned by os.cpu_count() is used.
        progress: whether to display a progress bar when caching for the first epoch.
        shuffle: whether to shuffle the whole data list before preparing the cache content for first epoch.
        seed: random seed if shuffle is `True`, default to `0`.
    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        replace_rate: float,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_init_workers: Optional[int] = None,
        num_replace_workers: Optional[int] = None,
        progress: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if shuffle:
            self.set_random_state(seed=seed)
            self.randomize(data)

        super().__init__(data, transform, cache_num, cache_rate, num_init_workers, progress)
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

        self._start_pos: int = 0
        self._update_lock: threading.Lock = threading.Lock()
        self._round: int = 1
        self._replace_done: bool = False
        self._replace_mgr: Optional[threading.Thread] = None

        self._compute_data_idx()

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
        return min((len(dataset) for dataset in self.data))

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
                AddChannel(),
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
                AddChannel(),
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
        transform: Optional[Callable] = None,
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

        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data
