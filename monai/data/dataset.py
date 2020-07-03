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
import sys
import threading
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset as _TorchDataset

from monai.transforms import apply_transform, Compose, Randomizable, Transform
from monai.utils import get_seed, progress_bar


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

    def __init__(self, data, transform: Optional[Callable] = None):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
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
    """

    def __init__(self, data, transform: Optional[Callable] = None, cache_dir=None):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            transform: transforms to execute operations on input data.
            cache_dir (Path or str or None): If specified, this is the location for persistent storage
                of pre-computed transformed data tensors. The cache_dir is computed once, and
                persists on disk until explicitly removed.  Different runs, programs, experiments
                may share a common cache dir provided that the transforms pre-processing is
                consistent.
        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

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
                cache_dir_path: Path = Path(self.cache_dir)
                if cache_dir_path.is_dir():
                    # TODO: Find way to hash transforms content as part of the cache
                    data_item_md5 = hashlib.md5(
                        json.dumps(item_transformed, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    hashfile: Path = Path(cache_dir_path) / f"{data_item_md5}.pt"

            if hashfile is not None and hashfile.is_file():
                item_transformed = torch.load(hashfile)
            else:
                item_transformed = self._pre_first_random_transform(item_transformed)
                if hashfile is not None:
                    # add sentinel flag to indicate that the transforms have already been computed.
                    item_transformed["cache"] = True
                    # NOTE: Writing to ".temp_write_cache" and then using a nearly atomic rename operation
                    #       to make the cache more robust to manual killing of parent process
                    #       which may leave partially written cache files in an incomplete state
                    temp_hash_file: Path = hashfile.with_suffix(".temp_write_cache")
                    torch.save(item_transformed, temp_hash_file)
                    temp_hash_file.rename(hashfile)

        return item_transformed

    def __getitem__(self, index):
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
        self, data, transform: Callable, cache_num: int = sys.maxsize, cache_rate: float = 1.0, num_workers: int = 0
    ):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
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
        self.cache_num = min(cache_num, int(len(self) * cache_rate), len(self))
        if self.cache_num > 0:
            self._cache = [None] * self.cache_num
            if num_workers > 0:
                self._item_processed = 0
                self._thread_lock = threading.Lock()
                with ThreadPool(num_workers) as p:
                    p.map(
                        self._load_cache_item_thread,
                        [(i, data[i], transform.transforms) for i in range(self.cache_num)],
                    )
            else:
                for i in range(self.cache_num):
                    self._cache[i] = self._load_cache_item(data[i], transform.transforms)
                    progress_bar(i + 1, self.cache_num, "Load and cache transformed data: ")

    def _load_cache_item(self, item, transforms):
        for _transform in transforms:
            # execute all the deterministic transforms
            if isinstance(_transform, Randomizable) or not isinstance(_transform, Transform):
                break
            item = apply_transform(_transform, item)
        return item

    def _load_cache_item_thread(self, args):
        i, item, transforms = args
        self._cache[i] = self._load_cache_item(item, transforms)
        with self._thread_lock:
            self._item_processed += 1
            progress_bar(self._item_processed, self.cache_num, "Load and cache transformed data: ")

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

    def __init__(self, datasets, transform: Optional[Callable] = None):
        """
        Args:
            datasets (list or tuple): list of datasets to zip together.
            transform: a callable data transform operates on the zipped item from `datasets`.
        """
        super().__init__(list(datasets), transform=transform)

    def __len__(self):
        return min([len(dataset) for dataset in self.data])

    def __getitem__(self, index: int):
        def to_list(x):
            return list(x) if isinstance(x, (tuple, list)) else [x]

        data = list()
        for dataset in self.data:
            data.extend(to_list(dataset[index]))
        if self.transform is not None:
            data = apply_transform(self.transform, data, map_items=False)  # transform the list data
        return data


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
        img,
        img_transform: Optional[Callable] = None,
        seg=None,
        seg_transform: Optional[Callable] = None,
        labels=None,
        label_transform: Optional[Callable] = None,
    ):
        """
        Initializes the dataset with the filename lists. The transform `img_transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            img (Sequence): sequence of images.
            img_transform: transform to apply to each element in `img`.
            seg (Sequence, optional): sequence of segmentations.
            seg_transform: transform to apply to each element in `seg`.
            labels (Sequence, optional): sequence of labels.
            label_transform: transform to apply to each element in `labels`.

        """
        items = [(img, img_transform), (seg, seg_transform), (labels, label_transform)]
        self.set_random_state(seed=get_seed())
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed

    def __len__(self):
        return len(self.dataset)

    def randomize(self):
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
