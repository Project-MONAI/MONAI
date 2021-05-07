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

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import IterableDataset

from monai.data.dataset import Dataset
from monai.data.utils import iter_patch
from monai.transforms import apply_transform
from monai.utils import NumpyPadMode, ensure_tuple

__all__ = ["PatchDataset", "GridPatchDataset", "PatchIter"]


class PatchIter:
    """
    A class to return a patch generator with predefined properties such as `patch_size`.
    Typically used with :py:class:`monai.data.GridPatchDataset`.
    """

    def __init__(
        self,
        patch_size: Sequence[int],
        start_pos: Sequence[int] = (),
        mode: Union[NumpyPadMode, str] = NumpyPadMode.WRAP,
        **pad_opts: Dict,
    ):
        """

        Args:
            patch_size: size of patches to generate slices for, 0/None selects whole dimension
            start_pos: starting position in the array, default is 0 for each dimension
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"wrap"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            pad_opts: padding options, see numpy.pad

        Note:
            The `patch_size` is the size of the
            patch to sample from the input arrays. It is assumed the arrays first dimension is the channel dimension which
            will be yielded in its entirety so this should not be specified in `patch_size`. For example, for an input 3D
            array with 1 channel of size (1, 20, 20, 20) a regular grid sampling of eight patches (1, 10, 10, 10) would be
            specified by a `patch_size` of (10, 10, 10).

        """
        self.patch_size = (None,) + tuple(patch_size)
        self.start_pos = ensure_tuple(start_pos)
        self.mode: NumpyPadMode = NumpyPadMode(mode)
        self.pad_opts = pad_opts

    def __call__(self, array):
        """
        Args:
            array: the image to generate patches from.
        """
        yield from iter_patch(
            array,
            patch_size=self.patch_size,  # expand to have the channel dim
            start_pos=self.start_pos,
            copy_back=False,
            mode=self.mode,
            **self.pad_opts,
        )


class GridPatchDataset(IterableDataset):
    """
    Yields patches from images read from an image dataset.
    Typically used with `PatchIter` so that the patches are chosen in a contiguous grid sampling scheme.

     .. code-block:: python

        import numpy as np

        from monai.data import GridPatchDataset, DataLoader, PatchIter
        from monai.transforms import RandShiftIntensity

        # image-level dataset
        images = [np.arange(16, dtype=float).reshape(1, 4, 4),
                  np.arange(16, dtype=float).reshape(1, 4, 4)]
        # image-level patch generator, "grid sampling"
        patch_iter = PatchIter(patch_size=(2, 2), start_pos=(0, 0))
        # patch-level intensity shifts
        patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)

        # construct the dataset
        ds = GridPatchDataset(dataset=images,
                              patch_iter=patch_iter,
                              transform=patch_intensity)
        # use the grid patch dataset
        for item in DataLoader(ds, batch_size=2, num_workers=2):
            print("patch size:", item[0].shape)
            print("coordinates:", item[1])

        # >>> patch size: torch.Size([2, 1, 2, 2])
        #     coordinates: tensor([[[0, 1], [0, 2], [0, 2]],
        #                          [[0, 1], [2, 4], [0, 2]]])

    """

    def __init__(
        self,
        dataset: Sequence,
        patch_iter: Callable,
        transform: Optional[Callable] = None,
        with_coordinates: bool = True,
    ) -> None:
        """
        Initializes this dataset in terms of the image dataset, patch generator, and an optional transform.

        Args:
            dataset: the dataset to read image data from.
            patch_iter: converts an input image (item from dataset) into a iterable of image patches.
                `patch_iter(dataset[idx])` must yield a tuple: (patches, coordinates).
                see also: :py:class:`monai.data.PatchIter`.
            transform: a callable data transform operates on the patches.
            with_coordinates: whether to yield the coordinates of each patch, default to `True`.

        """

        self.dataset = dataset
        self.patch_iter = patch_iter
        self.transform = transform
        self.with_coordinates = with_coordinates

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        iter_start, iter_end = 0, 1
        try:
            iter_end = len(self.dataset)  # TODO: support iterable self.dataset
        except TypeError:
            raise NotImplementedError("image dataset must implement `len()`.")

        if worker_info is not None:
            # split workload
            per_worker = int(np.ceil((iter_end - iter_start) / float(worker_info.num_workers)))
            iter_start = iter_start + worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, iter_end)

        for index in range(iter_start, iter_end):
            image = self.dataset[index]
            if not self.with_coordinates:
                for patch, *_ in self.patch_iter(image):  # patch_iter to yield at least 1 item: patch
                    out_patch = (
                        patch if self.transform is None else apply_transform(self.transform, patch, map_items=False)
                    )
                    yield out_patch
            else:
                for patch, slices, *_ in self.patch_iter(image):  # patch_iter to yield at least 2 items: patch, coords
                    out_patch = (
                        patch if self.transform is None else apply_transform(self.transform, patch, map_items=False)
                    )
                    yield out_patch, slices


class PatchDataset(Dataset):
    """
    returns a patch from an image dataset.
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
        self, dataset: Sequence, patch_func: Callable, samples_per_image: int = 1, transform: Optional[Callable] = None
    ) -> None:
        """
        Args:
            dataset: an image dataset to extract patches from.
            patch_func: converts an input image (item from dataset) into a sequence of image patches.
                patch_func(dataset[idx]) must return a sequence of patches (length `samples_per_image`).
            samples_per_image: `patch_func` should return a sequence of `samples_per_image` elements.
            transform: transform applied to each patch.
        """
        super().__init__(data=dataset, transform=transform)

        self.patch_func = patch_func
        if samples_per_image <= 0:
            raise ValueError("sampler_per_image must be a positive integer.")
        self.samples_per_image = int(samples_per_image)

    def __len__(self) -> int:
        return len(self.data) * self.samples_per_image

    def _transform(self, index: int):
        image_id = int(index / self.samples_per_image)
        image = self.data[image_id]
        patches = self.patch_func(image)
        if len(patches) != self.samples_per_image:
            raise RuntimeWarning(
                f"`patch_func` must return a sequence of length: samples_per_image={self.samples_per_image}."
            )
        patch_id = (index - image_id * self.samples_per_image) * (-1 if index < 0 else 1)
        patch = patches[patch_id]
        if self.transform is not None:
            patch = apply_transform(self.transform, patch, map_items=False)
        return patch
