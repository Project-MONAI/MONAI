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

import json
import logging
import os
import sys
from typing import Callable, Dict, List, Sequence, Union

import numpy as np

from monai.apps.datasets import DecathlonDataset
from monai.transforms import AsChannelFirstd, Compose, LoadImaged, Orientationd, Spacingd
from monai.utils import GridSampleMode


# TODO:: Test basic functionality
# TODO:: Unit Test
class DeepgrowDataset(DecathlonDataset):
    def __init__(
        self,
        dimension: int,
        pixdim: Sequence[float],
        root_dir: str,
        task: str,
        section: str,
        transform: Union[Sequence[Callable], Callable] = (),
        download: bool = False,
        seed: int = 0,
        val_frac: float = 0.2,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        limit: int = 0,
    ) -> None:
        self.dimension = dimension
        self.pixdim = pixdim
        self.limit = limit

        super().__init__(
            root_dir=root_dir,
            task=task,
            section=section,
            transform=transform,
            download=download,
            seed=seed,
            val_frac=val_frac,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def _generate_data_list(self, dataset_dir: str) -> List[Dict]:
        dataset = super()._generate_data_list(dataset_dir)

        tmp_dataset_dir = dataset_dir + "_{}.deep".format(self.section)
        new_datalist = create_dataset(
            datalist=dataset,
            keys=["image", "label"],
            output_dir=tmp_dataset_dir,
            dimension=self.dimension,
            pixdim=self.pixdim,
            limit=self.limit,
            relative_path=False,
        )

        dataset_json = os.path.join(tmp_dataset_dir, "dataset.json")
        with open(dataset_json, "w") as fp:
            json.dump({self.section: new_datalist}, fp, indent=2)
        return new_datalist


def _get_transforms(keys, pixdim):
    mode = [GridSampleMode.BILINEAR, GridSampleMode.NEAREST] if len(keys) == 2 else [GridSampleMode.BILINEAR]
    transforms = [
        LoadImaged(keys=keys),
        AsChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=pixdim, mode=mode),
        Orientationd(keys=keys, axcodes="RAS"),
    ]

    return Compose(transforms)


def _save_data_2d(vol_idx, data, keys, dataset_dir, relative_path):
    vol_image = data[keys[0]]
    vol_label = data.get(keys[1])
    data_list = []

    if len(vol_image.shape) == 4:
        logging.info("4D-Image, pick only first series; Image: {}; Label: {}".format(vol_image.shape, vol_label.shape))
        vol_image = vol_image[0]
        vol_image = np.moveaxis(vol_image, -1, 0)

    image_count = 0
    label_count = 0
    unique_labels_count = 0
    for sid in range(vol_image.shape[0]):
        image = vol_image[sid, ...]
        label = vol_label[sid, ...] if vol_label is not None else None

        if vol_label is not None and np.sum(label) == 0:
            continue

        image_file_prefix = "vol_idx_{:0>4d}_slice_{:0>3d}".format(vol_idx, sid)
        image_file = os.path.join(dataset_dir, "images", image_file_prefix)
        image_file += ".npy"

        os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
        np.save(image_file, image)
        image_count += 1

        # Test Data
        if vol_label is None:
            data_list.append(
                {
                    "image": image_file.replace(dataset_dir + "/", "") if relative_path else image_file,
                }
            )
            continue

        # For all Labels
        unique_labels = np.unique(label.flatten())
        unique_labels = unique_labels[unique_labels != 0]
        unique_labels_count = max(unique_labels_count, len(unique_labels))

        for idx in unique_labels:
            label_file_prefix = "{}_region_{:0>2d}".format(image_file_prefix, int(idx))
            label_file = os.path.join(dataset_dir, "labels", label_file_prefix)
            label_file += ".npy"

            os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)
            curr_label = (label == idx).astype(np.float32)
            np.save(label_file, curr_label)

            label_count += 1
            data_list.append(
                {
                    "image": image_file.replace(dataset_dir + "/", "") if relative_path else image_file,
                    "label": label_file.replace(dataset_dir + "/", "") if relative_path else label_file,
                    "region": int(idx),
                }
            )

    print(
        "{} => Image: {} => {}; Label: {} => {}; Unique Labels: {}".format(
            vol_idx,
            vol_image.shape,
            image_count,
            vol_label.shape if vol_label is not None else None,
            label_count,
            unique_labels_count,
        )
    )
    return data_list


def _save_data_3d(vol_idx, data, keys, dataset_dir, relative_path):
    vol_image = data[keys[0]]
    vol_label = data.get(keys[1])
    data_list = []

    if len(vol_image.shape) == 4:
        logging.info("4D-Image, pick only first series; Image: {}; Label: {}".format(vol_image.shape, vol_label.shape))
        vol_image = vol_image[0]
        vol_image = np.moveaxis(vol_image, -1, 0)

    image_count = 0
    label_count = 0
    unique_labels_count = 0

    image_file_prefix = "vol_idx_{:0>4d}".format(vol_idx)
    image_file = os.path.join(dataset_dir, "images", image_file_prefix)
    image_file += ".npy"

    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    np.save(image_file, vol_image)
    image_count += 1

    # Test Data
    if vol_label is None:
        data_list.append(
            {
                "image": image_file.replace(dataset_dir + "/", "") if relative_path else image_file,
            }
        )
    else:
        # For all Labels
        unique_labels = np.unique(vol_label.flatten())
        unique_labels = unique_labels[unique_labels != 0]
        unique_labels_count = max(unique_labels_count, len(unique_labels))

        for idx in unique_labels:
            label_file_prefix = "{}_region_{:0>2d}".format(image_file_prefix, int(idx))
            label_file = os.path.join(dataset_dir, "labels", label_file_prefix)
            label_file += ".npy"

            curr_label = (vol_label == idx).astype(np.float32)
            os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)
            np.save(label_file, curr_label)

            label_count += 1
            data_list.append(
                {
                    "image": image_file.replace(dataset_dir + "/", "") if relative_path else image_file,
                    "label": label_file.replace(dataset_dir + "/", "") if relative_path else label_file,
                    "region": int(idx),
                }
            )

    print(
        "{} => Image: {} => {}; Label: {} => {}; Unique Labels: {}".format(
            vol_idx,
            vol_image.shape,
            image_count,
            vol_label.shape if vol_label is not None else None,
            label_count,
            unique_labels_count,
        )
    )
    return data_list


def create_dataset(
    datalist, output_dir, dimension, pixdim, keys=("image", "label"), base_dir=None, limit=0, relative_path=False
) -> List[Dict]:
    if not isinstance(keys, list) and not isinstance(keys, tuple):
        keys = [keys]

    transforms = _get_transforms(keys, pixdim)
    new_datalist = []
    for idx in range(len(datalist)):
        if limit and idx >= limit:
            break

        image = datalist[idx][keys[0]]
        label = datalist[idx].get(keys[1]) if len(keys) > 1 else None
        if base_dir:
            image = os.path.join(base_dir, image)
            label = os.path.join(base_dir, label) if label else None

        image = os.path.abspath(image)
        label = os.path.abspath(label) if label else None

        print("{} => {}".format(image, label if label else None))
        if dimension == 2:
            data = _save_data_2d(
                vol_idx=idx,
                data=transforms({"image": image, "label": label}),
                keys=("image", "label"),
                dataset_dir=output_dir,
                relative_path=relative_path,
            )
        else:
            data = _save_data_3d(
                vol_idx=idx,
                data=transforms({"image": image, "label": label}),
                keys=("image", "label"),
                dataset_dir=output_dir,
                relative_path=relative_path,
            )
        new_datalist.extend(data)
    return new_datalist
