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

import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from monai.data import Dataset, SmartCacheDataset
from monai.data.image_reader import WSIReader

__all__ = ["PatchWSIDataset", "SmartCachePatchWSIDataset", "MaskedInferenceWSIDataset"]


class PatchWSIDataset(Dataset):
    """
    This dataset reads whole slide images, extracts regions, and creates patches.
    It also reads labels for each patch and provides each patch with its associated class labels.

    Args:
        data: the list of input samples including image, location, and label (see the note below for more details).
        region_size: the size of regions to be extracted from the whole slide image.
        grid_shape: the grid shape on which the patches should be extracted.
        patch_size: the size of patches extracted from the region on the grid.
        transform: transforms to be executed on input data.
        image_reader_name: the name of library to be used for loading whole slide imaging, either CuCIM or OpenSlide.
            Defaults to CuCIM.

    Note:
        The input data has the following form as an example:
        `[{"image": "path/to/image1.tiff", "location": [200, 500], "label": [0,0,0,1]}]`.

        This means from "image1.tiff" extract a region centered at the given location `location`
        with the size of `region_size`, and then extract patches with the size of `patch_size`
        from a square grid with the shape of `grid_shape`.
        Be aware the the `grid_shape` should construct a grid with the same number of element as `labels`,
        so for this example the `grid_shape` should be (2, 2).

    """

    def __init__(
        self,
        data: List,
        region_size: Union[int, Tuple[int, int]],
        grid_shape: Union[int, Tuple[int, int]],
        patch_size: int,
        transform: Optional[Callable] = None,
        image_reader_name: str = "cuCIM",
    ):
        super().__init__(data, transform)

        if isinstance(region_size, int):
            self.region_size = (region_size, region_size)
        else:
            self.region_size = region_size

        if isinstance(grid_shape, int):
            self.grid_shape = (grid_shape, grid_shape)
        else:
            self.grid_shape = grid_shape

        self.patch_size = patch_size
        self.sub_region_size = (self.region_size[0] / self.grid_shape[0], self.region_size[1] / self.grid_shape[1])

        self.image_path_list = list({x["image"] for x in self.data})
        self.image_reader_name = image_reader_name
        self.image_reader = WSIReader(image_reader_name)
        self.wsi_object_dict = None
        if self.image_reader_name != "openslide":
            # OpenSlide causes memory issue if we prefetch image objects
            self._fetch_wsi_objects()

    def _fetch_wsi_objects(self):
        """Load all the image objects and reuse them when asked for an item."""
        self.wsi_object_dict = {}
        for image_path in self.image_path_list:
            self.wsi_object_dict[image_path] = self.image_reader.read(image_path)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.image_reader_name == "openslide":
            img_obj = self.image_reader.read(sample["image"])
        else:
            img_obj = self.wsi_object_dict[sample["image"]]
        location = [sample["location"][i] - self.region_size[i] // 2 for i in range(len(self.region_size))]
        images, _ = self.image_reader.get_data(
            img=img_obj,
            location=location,
            size=self.region_size,
            grid_shape=self.grid_shape,
            patch_size=self.patch_size,
        )
        labels = np.array(sample["label"], dtype=np.float32)
        # expand dimensions to have 4 dimension as batch, class, height, and width.
        for _ in range(4 - labels.ndim):
            labels = np.expand_dims(labels, 1)
        patches = [{"image": images[i], "label": labels[i]} for i in range(len(sample["label"]))]
        if self.transform:
            patches = self.transform(patches)
        return patches


class SmartCachePatchWSIDataset(SmartCacheDataset):
    """Add SmartCache functionality to `PatchWSIDataset`.

    Args:
        data: the list of input samples including image, location, and label (see `PatchWSIDataset` for more details)
        region_size: the region to be extracted from the whole slide image.
        grid_shape: the grid shape on which the patches should be extracted.
        patch_size: the patches extracted from the region on the grid.
        image_reader_name: the name of library to be used for loading whole slide imaging, either CuCIM or OpenSlide.
            Defaults to CuCIM.
        transform: transforms to be executed on input data.
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

    """

    def __init__(
        self,
        data: List,
        region_size: Union[int, Tuple[int, int]],
        grid_shape: Union[int, Tuple[int, int]],
        patch_size: int,
        transform: Union[Sequence[Callable], Callable],
        image_reader_name: str = "cuCIM",
        replace_rate: float = 0.5,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_init_workers: Optional[int] = None,
        num_replace_workers: Optional[int] = None,
        progress: bool = True,
    ):
        patch_wsi_dataset = PatchWSIDataset(
            data=data,
            region_size=region_size,
            grid_shape=grid_shape,
            patch_size=patch_size,
            image_reader_name=image_reader_name,
        )
        super().__init__(
            data=patch_wsi_dataset,  # type: ignore
            transform=transform,
            replace_rate=replace_rate,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_init_workers=num_init_workers,
            num_replace_workers=num_replace_workers,
            progress=progress,
            shuffle=False,
        )


class MaskedInferenceWSIDataset(Dataset):
    """
    This dataset load the provided tissue masks at an arbitrary resolution level,
    and extract patches based on that mask from the associated whole slide image.

    Args:
        data: a list of sample including path to the mask and path to the whole slide image
            `[{"image": "path/to/image1.tiff", "label": "path/to/mask.npy}, ...]"`.
        patch_size: the size of patches to be extracted from the whole slide image for inference.
        transform: transforms to be executed on extracted patches.
        image_reader_name: the name of library to be used for loading whole slide imaging, either CuCIM or OpenSlide.
            Defaults to CuCIM.

    """

    def __init__(
        self,
        data: List,
        patch_size: Union[int, Tuple[int, int]],
        transform: Optional[Callable] = None,
        image_reader_name: str = "cuCIM",
    ) -> None:
        super().__init__(data, transform)

        if isinstance(patch_size, int):
            self.patch_size = np.array((patch_size, patch_size))
        else:
            self.patch_size = np.array(patch_size)
        self.image_reader_name = image_reader_name
        self.image_reader = WSIReader(image_reader_name)

        # process data and create a list of dictionaries containing all required data and metadata
        self.data_list = self._create_data_list(data)

        # calculate cummulative number of patches for all whole slide images
        self.cum_num_patches = np.cumsum([0] + [len(d["image_locations"]) for d in self.data_list])
        self.total_num_patches = self.cum_num_patches[-1]
        self.cum_num_patches = self.cum_num_patches[:-1]

    def _create_data_list(self, data: List[Dict]) -> List[Dict]:
        data_list = []
        print("Number of whole slide images: ", len(data))
        for sample in data:
            processed_data = self._preprocess_sample(sample)
            data_list.append(processed_data)
        return data_list

    def _preprocess_sample(self, sample: Dict) -> Dict:
        """
        Preprocess input data to load WSIReader object and the foreground mask,
        and define the locations where patches need to be extracted.
        """
        image = self.image_reader.read(sample["image"])
        mask = np.load(sample["label"]).T
        try:
            level, ratio = self._calculate_mask_level(image, mask)
        except ValueError as err:
            err.args = (sample["label"],) + err.args
            raise
        print(f"Mask ({sample['label']}) at level {int(level)}, with ratio {int(ratio)}")

        # get all indices for tissue region from the foreground mask
        # note: output same size as the foreground mask and not original wsi image size
        mask_locations = np.vstack(mask.nonzero()).T

        # convert mask locations to image locations to extract patches
        image_locations = (mask_locations + 0.5) * ratio - self.patch_size // 2

        return {
            "name": os.path.splitext(os.path.basename(sample["image"]))[0],
            "image": image,
            "mask_shape": mask.shape,
            "mask_locations": mask_locations.astype(int).tolist(),
            "image_locations": image_locations.astype(int).tolist(),
            "level": level,
        }

    def _calculate_mask_level(self, image, mask) -> Tuple[int, int]:
        """Calculate level of the mask and its ratio with respect to the whole slide image"""
        dim_y_img, dim_x_img, _ = image.shape
        dim_y_msk, dim_x_msk = mask.shape

        ratio_x = dim_x_img / dim_x_msk
        ratio_y = dim_y_img / dim_y_msk
        level_x = np.log2(ratio_x)

        if ratio_x != ratio_y:
            raise ValueError(
                "Image/Mask dimension does not match!"
                " dim_x_img / dim_x_msk : {} / {},"
                " dim_y_img / dim_y_msk : {} / {}".format(dim_x_img, dim_x_msk, dim_y_img, dim_y_msk)
            )
        elif not level_x.is_integer():
            raise ValueError(
                "Mask not at regular level (ratio not power of 2)," " image / mask ratio: {},".format(ratio_x)
            )
        return level_x, ratio_x

    def _load_a_sample(self, index: int) -> Dict:
        """
        Load sample given the index

        This method first, find the right patch to be extracted from the right whole slide image, and
        then load this patch and provide it with its image name and the corrsponding mask location.
        """
        sample_num = np.argmax(self.cum_num_patches > index) - 1
        sample = self.data_list[sample_num]
        patch_num = index - self.cum_num_patches[sample_num]
        image_location = sample["image_locations"][patch_num]
        mask_location = sample["mask_locations"][patch_num]

        image, _ = self.image_reader.get_data(
            img=sample["image"],
            location=image_location,
            size=self.patch_size.tolist(),
        )
        processed_sample = {"image": image, "name": sample["name"], "mask_location": mask_location}
        return processed_sample

    def __len__(self):
        return self.total_num_patches

    def __getitem__(self, index: int) -> List[Dict]:
        sample = [self._load_a_sample(index)]
        if self.transform:
            sample = self.transform(sample)
        return sample
