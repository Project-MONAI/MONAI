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

from copy import deepcopy
from typing import Callable, Dict, Hashable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from monai.config import KeysCollection
from monai.data.dataset import Dataset
from monai.data.iterable_dataset import IterableDataset
from monai.data.utils import iter_patch
from monai.transforms import apply_transform
from monai.utils import NumpyPadMode, deprecated_arg, ensure_tuple, first, look_up_option

__all__ = ["PatchDataset", "GridPatchDataset", "PatchIter", "PatchIterd"]


class PatchIter:
    """
    Return a patch generator with predefined properties such as `patch_size`.
    Typically used with :py:class:`monai.data.GridPatchDataset`.

    """

    def __init__(
        self, patch_size: Sequence[int], start_pos: Sequence[int] = (), mode: str = NumpyPadMode.WRAP, **pad_opts: Dict
    ):
        """

        Args:
            patch_size: size of patches to generate slices for, 0/None selects whole dimension
            start_pos: starting position in the array, default is 0 for each dimension
            mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
                ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                One of the listed string values or a user supplied function. Defaults to ``"wrap"``.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
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
        self.mode: NumpyPadMode = look_up_option(mode, NumpyPadMode)
        self.pad_opts = pad_opts

    def __call__(self, array: np.ndarray):
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
        mode: {``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``, ``"mean"``,
            ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            One of the listed string values or a user supplied function. Defaults to ``"wrap"``.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
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
        mode: str = NumpyPadMode.WRAP,
        **pad_opts,
    ):
        self.keys = ensure_tuple(keys)
        self.patch_iter = PatchIter(patch_size=patch_size, start_pos=start_pos, mode=mode, **pad_opts)

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
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

    .. deprecated:: 0.8.0
        ``dataset`` is deprecated, use ``data`` instead.

    """

    @deprecated_arg(name="dataset", new_name="data", since="0.8", msg_suffix="please use `data` instead.")
    def __init__(
        self,
        data: Union[Iterable, Sequence],
        patch_iter: Callable,
        transform: Optional[Callable] = None,
        with_coordinates: bool = True,
    ) -> None:
        super().__init__(data=data, transform=None)
        self.patch_iter = patch_iter
        self.patch_transform = transform
        self.with_coordinates = with_coordinates

    def __iter__(self):
        for image in super().__iter__():
            for patch, *others in self.patch_iter(image):
                out_patch = patch
                if self.patch_transform is not None:
                    out_patch = apply_transform(self.patch_transform, patch, map_items=False)
                if self.with_coordinates and len(others) > 0:  # patch_iter to yield at least 2 items: patch, coords
                    yield out_patch, others[0]
                else:
                    yield out_patch


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

    .. deprecated:: 0.8.0
        ``dataset`` is deprecated, use ``data`` instead.

    """

    @deprecated_arg(name="dataset", new_name="data", since="0.8", msg_suffix="please use `data` instead.")
    def __init__(
        self, data: Sequence, patch_func: Callable, samples_per_image: int = 1, transform: Optional[Callable] = None
    ) -> None:
        """
        Args:
            data: an image dataset to extract patches from.
            patch_func: converts an input image (item from dataset) into a sequence of image patches.
                patch_func(dataset[idx]) must return a sequence of patches (length `samples_per_image`).
            samples_per_image: `patch_func` should return a sequence of `samples_per_image` elements.
            transform: transform applied to each patch.
        """
        super().__init__(data=data, transform=transform)

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
