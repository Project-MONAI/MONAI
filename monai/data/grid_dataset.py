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

import sys
import warnings
from collections.abc import Callable, Generator, Hashable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from multiprocessing.managers import ListProxy
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING

import numpy as np
import torch

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayTensor
from monai.data.iterable_dataset import IterableDataset
from monai.data.utils import iter_patch, pickle_hashing
from monai.transforms import Compose, RandomizableTrait, Transform, apply_transform, convert_to_contiguous
from monai.utils import NumpyPadMode, ensure_tuple, first, min_version, optional_import

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

__all__ = ["PatchDataset", "GridPatchDataset", "PatchIter", "PatchIterd"]


class PatchIter:
    """
    Return a patch generator with predefined properties such as `patch_size`.
    Typically used with :py:class:`monai.data.GridPatchDataset`.

    """

    def __init__(
        self,
        patch_size: Sequence[int],
        start_pos: Sequence[int] = (),
        mode: str | None = NumpyPadMode.WRAP,
        **pad_opts: dict,
    ):
        """

        Args:
            patch_size: size of patches to generate slices for, 0/None selects whole dimension
            start_pos: starting position in the array, default is 0 for each dimension
            mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function.
                If None, no wrapping is performed. Defaults to ``"wrap"``.
                See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
                requires pytorch >= 1.10 for best compatibility.
            pad_opts: other arguments for the `np.pad` function.
                note that `np.pad` treats channel dimension as the first dimension.

        Note:
            The `patch_size` is the size of the
            patch to sample from the input arrays. It is assumed the arrays first dimension is the channel dimension which
            will be yielded in its entirety so this should not be specified in `patch_size`. For example, for an input 3D
            array with 1 channel of size (1, 20, 20, 20) a regular grid sampling of eight patches (1, 10, 10, 10) would be
            specified by a `patch_size` of (10, 10, 10).

        """
        self.patch_size = (None,) + tuple(patch_size)  # expand to have the channel dim
        self.start_pos = ensure_tuple(start_pos)
        self.mode = mode
        self.pad_opts = pad_opts

    def __call__(self, array: NdarrayTensor) -> Generator[tuple[NdarrayTensor, np.ndarray], None, None]:
        """
        Args:
            array: the image to generate patches from.

        """
        yield from iter_patch(
            array,
            patch_size=self.patch_size,  # type: ignore
            start_pos=self.start_pos,
            overlap=0.0,
            copy_back=False,
            mode=self.mode,
            **self.pad_opts,
        )


class PatchIterd:
    """
    Dictionary-based wrapper of :py:class:`monai.data.PatchIter`.
    Return a patch generator for dictionary data and the coordinate, Typically used
    with :py:class:`monai.data.GridPatchDataset`.
    Suppose all the expected fields specified by `keys` have same shape.

    Args:
        keys: keys of the corresponding items to iterate patches.
        patch_size: size of patches to generate slices for, 0/None selects whole dimension
        start_pos: starting position in the array, default is 0 for each dimension
        mode: available modes: (Numpy) {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            (PyTorch) {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function.
            If None, no wrapping is performed. Defaults to ``"wrap"``.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            requires pytorch >= 1.10 for best compatibility.
        pad_opts: other arguments for the `np.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    coords_key = "patch_coords"
    original_spatial_shape_key = "original_spatial_shape"
    start_pos_key = "start_pos"

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: Sequence[int],
        start_pos: Sequence[int] = (),
        mode: str | None = NumpyPadMode.WRAP,
        **pad_opts,
    ):
        self.keys = ensure_tuple(keys)
        self.patch_iter = PatchIter(patch_size=patch_size, start_pos=start_pos, mode=mode, **pad_opts)

    def __call__(
        self, data: Mapping[Hashable, NdarrayTensor]
    ) -> Generator[tuple[Mapping[Hashable, NdarrayTensor], np.ndarray], None, None]:
        d = dict(data)
        original_spatial_shape = d[first(self.keys)].shape[1:]

        for patch in zip(*[self.patch_iter(d[key]) for key in self.keys]):
            coords = patch[0][1]  # use the coordinate of the first item
            ret = {k: v[0] for k, v in zip(self.keys, patch)}
            # fill in the extra keys with unmodified data
            for k in set(d.keys()).difference(set(self.keys)):
                ret[k] = deepcopy(d[k])
            # also store the `coordinate`, `spatial shape of original image`, `start position` in the dictionary
            ret[self.coords_key] = coords
            ret[self.original_spatial_shape_key] = original_spatial_shape
            ret[self.start_pos_key] = self.patch_iter.start_pos
            yield ret, coords


class GridPatchDataset(IterableDataset):
    """
    Yields patches from data read from an image dataset.
    Typically used with `PatchIter` or `PatchIterd` so that the patches are chosen in a contiguous grid sampling scheme.

     .. code-block:: python

        import numpy as np

        from monai.data import GridPatchDataset, DataLoader, PatchIter, RandShiftIntensity

        # image-level dataset
        images = [np.arange(16, dtype=float).reshape(1, 4, 4),
                  np.arange(16, dtype=float).reshape(1, 4, 4)]
        # image-level patch generator, "grid sampling"
        patch_iter = PatchIter(patch_size=(2, 2), start_pos=(0, 0))
        # patch-level intensity shifts
        patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)

        # construct the dataset
        ds = GridPatchDataset(data=images,
                              patch_iter=patch_iter,
                              transform=patch_intensity)
        # use the grid patch dataset
        for item in DataLoader(ds, batch_size=2, num_workers=2):
            print("patch size:", item[0].shape)
            print("coordinates:", item[1])

        # >>> patch size: torch.Size([2, 1, 2, 2])
        #     coordinates: tensor([[[0, 1], [0, 2], [0, 2]],
        #                          [[0, 1], [2, 4], [0, 2]]])

    Args:
        data: the data source to read image data from.
        patch_iter: converts an input image (item from dataset) into a iterable of image patches.
            `patch_iter(dataset[idx])` must yield a tuple: (patches, coordinates).
            see also: :py:class:`monai.data.PatchIter` or :py:class:`monai.data.PatchIterd`.
        transform: a callable data transform operates on the patches.
        with_coordinates: whether to yield the coordinates of each patch, default to `True`.
        cache: whether to use cache mache mechanism, default to `False`.
            see also: :py:class:`monai.data.CacheDataset`.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads if computing cache in the initialization.
            If num_workers is None then the number returned by os.cpu_count() is used.
            If a value less than 1 is specified, 1 will be used instead.
        progress: whether to display a progress bar.
        copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
            default to `True`. if the random transforms don't modify the cached content
            (for example, randomly crop from the cached image and deepcopy the crop region)
            or if every cache item is only used once in a `multi-processing` environment,
            may set `copy=False` for better performance.
        as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
            it may help improve the performance of following logic.
        hash_func: a callable to compute hash from data items to be cached.
            defaults to `monai.data.utils.pickle_hashing`.

    """

    def __init__(
        self,
        data: Iterable | Sequence,
        patch_iter: Callable,
        transform: Callable | None = None,
        with_coordinates: bool = True,
        cache: bool = False,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int | None = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        hash_func: Callable[..., bytes] = pickle_hashing,
    ) -> None:
        super().__init__(data=data, transform=None)
        if transform is not None and not isinstance(transform, Compose):
            transform = Compose(transform)
        self.patch_iter = patch_iter
        self.patch_transform = transform
        self.with_coordinates = with_coordinates
        self.set_num = cache_num
        self.set_rate = cache_rate
        self.progress = progress
        self.copy_cache = copy_cache
        self.as_contiguous = as_contiguous
        self.hash_func = hash_func
        self.num_workers = num_workers
        if self.num_workers is not None:
            self.num_workers = max(int(self.num_workers), 1)
        self._cache: list | ListProxy = []
        self._cache_other: list | ListProxy = []
        self.cache = cache
        self.first_random: int | None = None
        if self.patch_transform is not None:
            self.first_random = self.patch_transform.get_index_of_first(
                lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
            )

        if self.cache:
            if isinstance(data, Iterator):
                raise TypeError("Data can not be iterator when cache is True")
            self.set_data(data)  # type: ignore

    def set_data(self, data: Sequence) -> None:
        """
        Set the input data and run deterministic transforms to generate cache content.

        Note: should call this func after an entire epoch and must set `persistent_workers=False`
        in PyTorch DataLoader, because it needs to create new worker processes based on new
        generated cache content.

        """
        self.data = data

        # only compute cache for the unique items of dataset, and record the last index for duplicated items
        mapping = {self.hash_func(v): i for i, v in enumerate(self.data)}
        self.cache_num = min(int(self.set_num), int(len(mapping) * self.set_rate), len(mapping))
        self._hash_keys = list(mapping)[: self.cache_num]
        indices = list(mapping.values())[: self.cache_num]
        self._cache, self._cache_other = zip(*self._fill_cache(indices))  # type: ignore

    def _fill_cache(self, indices=None) -> list:
        """
        Compute and fill the cache content from data source.

        Args:
            indices: target indices in the `self.data` source to compute cache.
                if None, use the first `cache_num` items.

        """
        if self.cache_num <= 0:
            return []
        if indices is None:
            indices = list(range(self.cache_num))
        if self.progress and not has_tqdm:
            warnings.warn("tqdm is not installed, will not show the caching progress bar.")

        pfunc = tqdm if self.progress and has_tqdm else (lambda v, **_: v)
        with ThreadPool(self.num_workers) as p:
            return list(pfunc(p.imap(self._load_cache_item, indices), total=len(indices), desc="Loading dataset"))

    def _load_cache_item(self, idx: int):
        """
        Args:
            idx: the index of the input data sequence.
        """
        item = self.data[idx]  # type: ignore
        patch_cache, other_cache = [], []
        for patch, *others in self.patch_iter(item):
            if self.first_random is not None:
                patch = self.patch_transform(patch, end=self.first_random, threading=True)  # type: ignore

            if self.as_contiguous:
                patch = convert_to_contiguous(patch, memory_format=torch.contiguous_format)
            if self.with_coordinates and len(others) > 0:  # patch_iter to yield at least 2 items: patch, coords
                other_cache.append(others[0])
            patch_cache.append(patch)
        return patch_cache, other_cache

    def _generate_patches(self, src, **apply_args):
        """
        yield patches optionally post-processed by transform.

        Args:
            src: a iterable of image patches.
            apply_args: other args for `self.patch_transform`.

        """
        for patch, *others in src:
            out_patch = patch
            if self.patch_transform is not None:
                out_patch = self.patch_transform(patch, **apply_args)
            if self.with_coordinates and len(others) > 0:  # patch_iter to yield at least 2 items: patch, coords
                yield out_patch, others[0]
            else:
                yield out_patch

    def __iter__(self):
        if self.cache:
            cache_index = None
            for image in super().__iter__():
                key = self.hash_func(image)
                if key in self._hash_keys:
                    # if existing in cache, try to get the index in cache
                    cache_index = self._hash_keys.index(key)
                if cache_index is None:
                    # no cache for this index, execute all the transforms directly
                    yield from self._generate_patches(self.patch_iter(image))
                else:
                    if self._cache is None:
                        raise RuntimeError(
                            "Cache buffer is not initialized, please call `set_data()` before epoch begins."
                        )
                    data = self._cache[cache_index]
                    other = self._cache_other[cache_index]

                    # load data from cache and execute from the first random transform
                    data = deepcopy(data) if self.copy_cache else data
                    yield from self._generate_patches(zip(data, other), start=self.first_random)
        else:
            for image in super().__iter__():
                yield from self._generate_patches(self.patch_iter(image))


class PatchDataset(IterableDataset):
    """
    Yields patches from data read from an image dataset.
    The patches are generated by a user-specified callable `patch_func`,
    and are optionally post-processed by `transform`.
    For example, to generate random patch samples from an image dataset:

    .. code-block:: python

        import numpy as np

        from monai.data import PatchDataset, DataLoader
        from monai.transforms import RandSpatialCropSamples, RandShiftIntensity

        # image dataset
        images = [np.arange(16, dtype=float).reshape(1, 4, 4),
                  np.arange(16, dtype=float).reshape(1, 4, 4)]
        # image patch sampler
        n_samples = 5
        sampler = RandSpatialCropSamples(roi_size=(3, 3), num_samples=n_samples,
                                         random_center=True, random_size=False)
        # patch-level intensity shifts
        patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)
        # construct the patch dataset
        ds = PatchDataset(dataset=images,
                          patch_func=sampler,
                          samples_per_image=n_samples,
                          transform=patch_intensity)

        # use the patch dataset, length: len(images) x samplers_per_image
        print(len(ds))

        >>> 10

        for item in DataLoader(ds, batch_size=2, shuffle=True, num_workers=2):
            print(item.shape)

        >>> torch.Size([2, 1, 3, 3])

    """

    def __init__(
        self, data: Sequence, patch_func: Callable, samples_per_image: int = 1, transform: Callable | None = None
    ) -> None:
        """
        Args:
            data: an image dataset to extract patches from.
            patch_func: converts an input image (item from dataset) into a sequence of image patches.
                patch_func(dataset[idx]) must return a sequence of patches (length `samples_per_image`).
            samples_per_image: `patch_func` should return a sequence of `samples_per_image` elements.
            transform: transform applied to each patch.
        """
        super().__init__(data=data, transform=None)

        self.patch_func = patch_func
        if samples_per_image <= 0:
            raise ValueError("sampler_per_image must be a positive integer.")
        self.samples_per_image = int(samples_per_image)
        self.patch_transform = transform

    def __len__(self) -> int:
        return len(self.data) * self.samples_per_image  # type: ignore

    def __iter__(self):
        for image in super().__iter__():
            patches = self.patch_func(image)
            if len(patches) != self.samples_per_image:
                raise RuntimeWarning(
                    f"`patch_func` must return a sequence of length: samples_per_image={self.samples_per_image}."
                )
            for patch in patches:
                out_patch = patch
                if self.patch_transform is not None:
                    out_patch = apply_transform(self.patch_transform, patch, map_items=False)
                yield out_patch
