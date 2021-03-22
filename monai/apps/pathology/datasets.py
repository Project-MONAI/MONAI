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
from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple

import numpy as np

from monai.data import Dataset, SmartCacheDataset
from monai.data.image_reader import WSIReader

__all__ = ["PatchWSIDataset", "SmartCachePatchWSIDataset"]


class PatchWSIDataset(Dataset):
    """
    This dataset read whole slide images, extract regions, and crate patches.
    It reads labels for each patch and privide each patch with its associated class labels.

    Args:
        data: The input image directory and the label file
        [{"image": "path/to/image/directory", "label": "path/to/label/file.txt"}]
        region_size: the region to be extracted from the whole slide image
        grid_size: the grid size on which the patches should be extracted
        patch_size: the patches extracted from the region on the grid
        image_reader_name: (cuCIM is default)
        transform:

    """

    def __init__(
        self,
        data: dict,
        region_size: Union[int, Tuple[int, int]],
        grid_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        image_reader_name="cuCIM",
        transform=None,
    ):
        if type(region_size) == int:
            self.region_size = (region_size, region_size)
        else:
            self.region_size = region_size

        if type(grid_size) == int:
            self.grid_size = (grid_size, grid_size)
        else:
            self.grid_size = grid_size

        self.patch_size = patch_size
        self.sub_region_size = (self.region_size[0] / self.grid_size[0], self.region_size[1] / self.grid_size[1])

        self.transform = transform
        self.image_base_path = data[0]["image"]
        self.samples = self.load_samples(data[0]["label"])
        self.image_path_list = {x[0] for x in self.samples}
        self.num_samples = len(self.samples)

        self.image_reader_name = image_reader_name
        self.image_reader = WSIReader(image_reader_name)
        self.wsi_object_dict = {}
        self._fetch_wsi_objects()

    def _fetch_wsi_objects(self):
        for image_path in self.image_path_list:
            self.wsi_object_dict[image_path] = self.image_reader.read(image_path)

    def process_label_row(self, row):
        row = row.strip("\n").split(",")
        # create full image path
        image_name = row[0] + ".tif"
        image_path = os.path.join(self.image_base_path, image_name)
        # change center locations to upper left location
        location = (int(row[2]) - self.region_size[1] // 2, int(row[1]) - self.region_size[0] // 2)
        # convert labels to float32 and add empty HxW channel to label
        labels = tuple(int(lbl) for lbl in row[3:])
        labels = np.array(labels, dtype=np.float32)[:, np.newaxis, np.newaxis]
        return image_path, location, labels

    def load_samples(self, loc_path):
        with open(loc_path) as label_file:
            rows = [self.process_label_row(row) for row in label_file.readlines()]
        return rows

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_path, location, labels = self.samples[index]
        # OpenSlide causes issue if using the stored image objects
        if self.image_reader_name == "openslide":
            img_obj = self.image_reader.read(image_path)
        else:
            img_obj = self.wsi_object_dict[image_path]
        images, _ = self.image_reader.get_data(
            img=img_obj,
            location=location,
            size=self.region_size,
            grid_shape=self.grid_size,
            patch_size=self.patch_size,
        )
        samples = [{"image": images[i], "label": labels[i]} for i in range(labels.shape[0])]
        if self.transform:
            samples = self.transform(samples)
        return samples


class SmartCachePatchWSIDataset(SmartCacheDataset):
    """
    Add SmartCache functionality to PatchWSIDataset
    """

    def __init__(
        self,
        data,
        region_size,
        grid_size,
        patch_size,
        transform,
        replace_rate,
        cache_num,
        cache_rate=1.0,
        num_init_workers=None,
        num_replace_workers=0,
        image_reader_name="cuCIM",
    ):
        extractor = PatchWSIDataset(data, region_size, grid_size, patch_size, image_reader_name)
        super().__init__(
            data=extractor,
            transform=transform,
            replace_rate=replace_rate,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_init_workers=num_init_workers,
            num_replace_workers=num_replace_workers,
        )
