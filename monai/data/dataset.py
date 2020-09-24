# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import math
import sys
import threading
import time
import warnings
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset as _TorchDataset

from monai.transforms import Compose, Randomizable, Transform, apply_transform
from monai.utils import get_seed, min_version, optional_import

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")


class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
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

    def __getitem__(self, index: int):
        data = self.data[index]
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data


class PersistentDataset(Dataset):
    """
    Persistent storage of pre-computed values to efficiently manage larger than memory dictionary format data,
    it can operate transforms for specific fields.  Results from the non-random transform components are computed
    when first used, and stored in the `cache_dir` for rapid retrieval on subsequent uses.

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]

    For a composite transform like

    .. code-block:: python

        [ LoadNiftid(keys=['image', 'label']),
          Orientationd(keys=['image', 'label'], axcodes='RAS'),
          ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
          RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96),
                                 pos=1, neg=1, num_samples=4, image_key='image', image_threshold=0),
          ToTensord(keys=['image', 'label'])]

    Upon first use a filename based dataset will be processed by the transform for the
    [LoadNiftid, Orientationd, ScaleIntensityRanged] and the resulting tensor written to
    the `cache_dir` before applying the remaining random dependant transforms
    [RandCropByPosNegLabeld, ToTensord] elements for use in the analysis.

    Subsequent uses of a dataset directly read pre-processed results from `cache_dir`
    followed by applying the random dependant parts of transform processing.

    Note:
        The input data must be a list of file paths and will hash them as cache keys.

    """

    def __init__(
        self,
        data: Sequence[str],
        transform: Union[Sequence[Callable], Callable],
        cache_dir: Optional[Union[Path, str]] = None,
    ) -> None:
        """
        Args:
            data: input data file paths to load and transform to generate dataset for model.
                `PersistentDataset` expects input data to be a list of file paths and hashes them as cache keys.
            transform: transforms to execute operations on input data.
            cache_dir: If specified, this is the location for persistent storage
                of pre-computed transformed data tensors. The cache_dir is computed once, and
                persists on disk until explicitly removed.  Different runs, programs, experiments
                may share a common cache dir provided that the transforms pre-processing is consistent.
                If the cache_dir doesn't exist, will automatically create it.
        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True)
            if not self.cache_dir.is_dir():
                raise ValueError("cache_dir must be a directory.")

    def _pre_first_random_transform(self, item_transformed):
        """
        Process the data from original state up to the first random element.

        Args:
            item_transformed: The data to be transformed

        Returns:
            the transformed element up to the first identified
            random transform object
        """
        for _transform in self.transform.transforms:  # pytype: disable=attribute-error
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            item_transformed = apply_transform(_transform, item_transformed)
        return item_transformed

    def _first_random_and_beyond_transform(self, item_transformed):
        """
        Process the data from before the first random transform to the final state ready for evaluation.

        Args:
            item_transformed: The data to be transformed (already processed up to the first random transform)

        Returns:
            the transformed element through the random transforms
        """
        start_post_randomize_run = False
        for _transform in self.transform.transforms:  # pytype: disable=attribute-error
            if (
                start_post_randomize_run
                or isinstance(_transform, Randomizable)
                or not isinstance(_transform, Transform)
            ):
                start_post_randomize_run = True
                item_transformed = apply_transform(_transform, item_transformed)
        return item_transformed

    def _pre_first_random_cachecheck(self, item_transformed):
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
        if item_transformed.get("cached", False) is False:
            hashfile = None
            if self.cache_dir is not None:
                # TODO: Find way to hash transforms content as part of the cache
                data_item_md5 = hashlib.md5(json.dumps(item_transformed, sort_keys=True).encode("utf-8")).hexdigest()
                hashfile = self.cache_dir / f"{data_item_md5}.pt"

            if hashfile is not None and hashfile.is_file():
                item_transformed = torch.load(hashfile)
            else:
                item_transformed = self._pre_first_random_transform(item_transformed)
                if hashfile is not None:
                    # add sentinel flag to indicate that the transforms have already been computed.
                    item_transformed["cached"] = True
                    # NOTE: Writing to ".temp_write_cache" and then using a nearly atomic rename operation
                    #       to make the cache more robust to manual killing of parent process
                    #       which may leave partially written cache files in an incomplete state
                    temp_hash_file = hashfile.with_suffix(".temp_write_cache")
                    torch.save(item_transformed, temp_hash_file)
                    temp_hash_file.rename(hashfile)

        return item_transformed

    def __getitem__(self, index: int):
        pre_random_item = self._pre_first_random_cachecheck(self.data[index])
        post_random_item = self._first_random_and_beyond_transform(pre_random_item)
        return post_random_item


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

    For example, if the transform is a `Compose` of::

        transforms = Compose([
            LoadNiftid(),
            AddChanneld(),
            Spacingd(),
            Orientationd(),
            ScaleIntensityRanged(),
            RandCropByPosNegLabeld(),
            ToTensord()
        ])

    when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
    this dataset will cache the results up to ``ScaleIntensityRanged``, as
    all non-random transforms `LoadNiftid`, `AddChanneld`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
    can be cached. During training, the dataset will load the cached results and run
    ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
    and the outcome not cached.
    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
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
                If 0 a single thread will be used. Default is 0.
        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data, transform)
        self.cache_num = min(cache_num, int(len(data) * cache_rate), len(data))
        if self.cache_num > 0:
            self._cache = [None] * self.cache_num
            if has_tqdm:
                pbar = tqdm(total=self.cache_num, desc="Load and cache transformed data")
            else:
                warnings.warn("tqdm is not installed, will not show the caching progress bar.")
                pbar = None

            if num_workers > 0:
                self._item_processed = 0
                self._thread_lock = threading.Lock()
                with ThreadPool(num_workers) as p:
                    p.map(
                        self._load_cache_item_thread,
                        [(i, data[i], transform.transforms, pbar) for i in range(self.cache_num)],
                    )
            else:
                for i in range(self.cache_num):
                    self._cache[i] = self._load_cache_item(data[i], transform.transforms)
                    if pbar is not None:
                        pbar.update(1)
            if pbar is not None:
                pbar.close()

    def _load_cache_item(self, item: Any, transforms: Sequence[Callable]):
        """
        Args:
            item: input item to load and transform to generate dataset for model.
            transforms: transforms to execute operations on input item.
        """
        for _transform in transforms:
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            item = apply_transform(_transform, item)
        return item

    def _load_cache_item_thread(self, args: Any) -> None:
        """
        Args:
            args: tuple with contents (i, item, transforms, pbar).
                i: the index to load the cached item to.
                item: input item to load and transform to generate dataset for model.
                transforms: transforms to execute operations on input item.
                pbar: tqdm progress bar
        """
        i, item, transforms, pbar = args
        self._cache[i] = self._load_cache_item(item, transforms)
        if pbar is not None:
            with self._thread_lock:
                pbar.update(1)

    def __getitem__(self, index):
        if index < self.cache_num:
            # load data from cache and execute from the first random transform
            start_run = False
            data = self._cache[index]
            for _transform in self.transform.transforms:  # pytype: disable=attribute-error
                if not start_run and not isinstance(_transform, Randomizable) and isinstance(_transform, Transform):
                    continue
                else:
                    start_run = True
                data = apply_transform(_transform, data)
        else:
            # no cache for this data, execute all the transforms directly
            data = super(CacheDataset, self).__getitem__(index)
        return data


class SmartCacheDataset(CacheDataset):
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

    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        replace_rate: float,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_init_workers: int = 0,
        num_replace_workers: int = 0,
    ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: transforms to execute operations on input data.
            replace_rate: percentage of the cached items to be replaced in every epoch.
            cache_num: number of items to be cached. Default is `sys.maxsize`.
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            cache_rate: percentage of cached data in total, default is 1.0 (cache all).
                will take the minimum of (cache_num, data_length x cache_rate, data_length).
            num_init_workers: the number of worker threads to initialize the cache for first epoch.
                if 0, run in main thread, no separate thread will open.
            num_replace_workers: the number of worker threads to prepare the replacement cache for every epoch.
                if 0, run in main thread, no separate thread will open.
        """
        super().__init__(data, transform, cache_num, cache_rate, num_init_workers)
        if self.cache_num >= len(data):
            raise ValueError("cache_num must be smaller than dataset length to support replacement.")
        if replace_rate <= 0:
            raise ValueError("replace_rate must be greater than 0, otherwise, please use CacheDataset.")
        self.num_replace_workers: int = num_replace_workers

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
        return self._replace_mgr.isAlive()

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
            if self._replace_done:
                remain_num: int = self.cache_num - self._replace_num
                for i in range(remain_num):
                    self._cache[i] = self._cache[i + self._replace_num]
                for i in range(self._replace_num):
                    self._cache[remain_num + i] = self._replacements[i]

                self._start_pos += self._replace_num
                if self._start_pos >= self._total_num:
                    self._start_pos -= self._total_num

                self._compute_data_idx()

                # ready for next round
                self._round += 1
                self._replace_done = False
                return True
            else:
                return False

    def update_cache(self):
        """
        Update cache items for current epoch, need to call this function before every epoch.
        If the cache has been shutdown before, need to restart the `_replace_mgr` thread.

        """
        if not self._replace_mgr.isAlive():
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
            else:
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
        self._replacements[index] = self._load_cache_item(self.data[pos], self.transform.transforms)  # type: ignore

    def _compute_replacements(self):
        """
        Compute expected items for the replacement of next epoch, execute deterministic transforms.
        It can support multi-threads to accelerate the computation progress.

        """
        if self.num_replace_workers > 0:
            with ThreadPool(self.num_replace_workers) as p:
                p.map(self._replace_cache_thread, list(range(self._replace_num)))
        else:
            for i in range(self._replace_num):
                self._replace_cache_thread(i)
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

    def __getitem__(self, index: int):
        def to_list(x):
            return list(x) if isinstance(x, (tuple, list)) else [x]

        data = list()
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
                LoadNifti(image_only=True),
                AddChannel(),
                RandAdjustContrast()
            ]
        )
        ArrayDataset(img_file_list, img_transform=img_transform)

    If training based on images and the metadata, the array transforms can not be composed
    because several transforms receives multiple parameters or return multiple values. Then Users need
    to define their own callable method to parse metadata from `LoadNifti` or set `affine` matrix
    to `Spacing` transform::

        class TestCompose(Compose):
            def __call__(self, input_):
                img, metadata = self.transforms[0](input_)
                img = self.transforms[1](img)
                img, _, _ = self.transforms[2](img, metadata["affine"])
                return self.transforms[3](img), metadata
        img_transform = TestCompose(
            [
                LoadNifti(image_only=False),
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
        self._seed = self.R.randint(np.iinfo(np.int32).max)

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
