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

import logging
import os
from typing import Dict, List

import numpy as np

from monai.transforms import Compose, EnsureChannelFirstd, FromMetaTensord, LoadImaged, Orientationd, Spacingd, ToNumpyd
from monai.utils import GridSampleMode
from monai.utils.enums import PostFix


def create_dataset(
    datalist,
    output_dir: str,
    dimension: int,
    pixdim,
    image_key: str = "image",
    label_key: str = "label",
    base_dir=None,
    limit: int = 0,
    relative_path: bool = False,
    transforms=None,
) -> List[Dict]:
    """
    Utility to pre-process and create dataset list for Deepgrow training over on existing one.
    The input data list is normally a list of images and labels (3D volume) that needs pre-processing
    for Deepgrow training pipeline.

    Args:
        datalist: A list of data dictionary. Each entry should at least contain 'image_key': <image filename>.
            For example, typical input data can be a list of dictionaries::

                [{'image': <image filename>, 'label': <label filename>}]

        output_dir: target directory to store the training data for Deepgrow Training
        pixdim: output voxel spacing.
        dimension: dimension for Deepgrow training.  It can be 2 or 3.
        image_key: image key in input datalist. Defaults to 'image'.
        label_key: label key in input datalist. Defaults to 'label'.
        base_dir: base directory in case related path is used for the keys in datalist.  Defaults to None.
        limit: limit number of inputs for pre-processing.  Defaults to 0 (no limit).
        relative_path: output keys values should be based on relative path.  Defaults to False.
        transforms: explicit transforms to execute operations on input data.

    Raises:
        ValueError: When ``dimension`` is not one of [2, 3]
        ValueError: When ``datalist`` is Empty

    Returns:
        A new datalist that contains path to the images/labels after pre-processing.

    Example::

        datalist = create_dataset(
            datalist=[{'image': 'img1.nii', 'label': 'label1.nii'}],
            base_dir=None,
            output_dir=output_2d,
            dimension=2,
            image_key='image',
            label_key='label',
            pixdim=(1.0, 1.0),
            limit=0,
            relative_path=True
        )

        print(datalist[0]["image"], datalist[0]["label"])
    """

    if dimension not in [2, 3]:
        raise ValueError("Dimension can be only 2 or 3 as Deepgrow supports only 2D/3D Training")

    if not len(datalist):
        raise ValueError("Input datalist is empty")

    transforms = _default_transforms(image_key, label_key, pixdim) if transforms is None else transforms
    new_datalist = []
    for idx in range(len(datalist)):
        if limit and idx >= limit:
            break

        image = datalist[idx][image_key]
        label = datalist[idx].get(label_key, None)
        if base_dir:
            image = os.path.join(base_dir, image)
            label = os.path.join(base_dir, label) if label else None

        image = os.path.abspath(image)
        label = os.path.abspath(label) if label else None

        logging.info(f"Image: {image}; Label: {label if label else None}")
        data = transforms({image_key: image, label_key: label})
        if dimension == 2:
            data = _save_data_2d(
                vol_idx=idx,
                vol_image=data[image_key],
                vol_label=data[label_key],
                dataset_dir=output_dir,
                relative_path=relative_path,
            )
        else:
            data = _save_data_3d(
                vol_idx=idx,
                vol_image=data[image_key],
                vol_label=data[label_key],
                dataset_dir=output_dir,
                relative_path=relative_path,
            )
        new_datalist.extend(data)
    return new_datalist


def _default_transforms(image_key, label_key, pixdim):
    keys = [image_key] if label_key is None else [image_key, label_key]
    mode = [GridSampleMode.BILINEAR, GridSampleMode.NEAREST] if len(keys) == 2 else [GridSampleMode.BILINEAR]
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=pixdim, mode=mode),
            FromMetaTensord(keys=keys),
            ToNumpyd(keys=keys + [PostFix.meta(k) for k in keys]),
        ]
    )


def _save_data_2d(vol_idx, vol_image, vol_label, dataset_dir, relative_path):
    data_list = []
    if len(vol_image.shape) == 4:
        logging.info(
            "4D-Image, pick only first series; Image: {}; Label: {}".format(
                vol_image.shape, vol_label.shape if vol_label is not None else None
            )
        )
        vol_image = vol_image[0]
        vol_image = np.moveaxis(vol_image, -1, 0)
        if vol_label is not None:
            vol_label = vol_label[0]
            vol_label = np.moveaxis(vol_label, -1, 0)

    image_count = 0
    label_count = 0
    unique_labels_count = 0
    for sid in range(vol_image.shape[0]):
        image = vol_image[sid, ...]
        label = vol_label[sid, ...] if vol_label is not None else None

        if vol_label is not None and np.sum(label) == 0:
            continue

        image_file_prefix = f"vol_idx_{vol_idx:0>4d}_slice_{sid:0>3d}"
        image_file = os.path.join(dataset_dir, "images", image_file_prefix)
        image_file += ".npy"

        os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
        np.save(image_file, image)
        image_count += 1

        # Test Data
        if vol_label is None:
            data_list.append(
                {"image": image_file.replace(dataset_dir + os.pathsep, "") if relative_path else image_file}
            )
            continue

        # For all Labels
        unique_labels = np.unique(label.flatten())
        unique_labels = unique_labels[unique_labels != 0]
        unique_labels_count = max(unique_labels_count, len(unique_labels))

        for idx in unique_labels:
            label_file_prefix = f"{image_file_prefix}_region_{int(idx):0>2d}"
            label_file = os.path.join(dataset_dir, "labels", label_file_prefix)
            label_file += ".npy"

            os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)
            curr_label = (label == idx).astype(np.float32)
            np.save(label_file, curr_label)

            label_count += 1
            data_list.append(
                {
                    "image": image_file.replace(dataset_dir + os.pathsep, "") if relative_path else image_file,
                    "label": label_file.replace(dataset_dir + os.pathsep, "") if relative_path else label_file,
                    "region": int(idx),
                }
            )

    if unique_labels_count >= 20:
        logging.warning(f"Unique labels {unique_labels_count} exceeds 20. Please check if this is correct.")

    logging.info(
        "{} => Image Shape: {} => {}; Label Shape: {} => {}; Unique Labels: {}".format(
            vol_idx,
            vol_image.shape,
            image_count,
            vol_label.shape if vol_label is not None else None,
            label_count,
            unique_labels_count,
        )
    )
    return data_list


def _save_data_3d(vol_idx, vol_image, vol_label, dataset_dir, relative_path):
    data_list = []

    if len(vol_image.shape) == 4:
        logging.info(
            "4D-Image, pick only first series; Image: {}; Label: {}".format(
                vol_image.shape, vol_label.shape if vol_label is not None else None
            )
        )
        vol_image = vol_image[0]
        vol_image = np.moveaxis(vol_image, -1, 0)

    image_count = 0
    label_count = 0
    unique_labels_count = 0

    image_file_prefix = f"vol_idx_{vol_idx:0>4d}"
    image_file = os.path.join(dataset_dir, "images", image_file_prefix)
    image_file += ".npy"

    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    np.save(image_file, vol_image)
    image_count += 1

    # Test Data
    if vol_label is None:
        data_list.append({"image": image_file.replace(dataset_dir + os.pathsep, "") if relative_path else image_file})
    else:
        # For all Labels
        unique_labels = np.unique(vol_label.flatten())
        unique_labels = unique_labels[unique_labels != 0]
        unique_labels_count = max(unique_labels_count, len(unique_labels))

        for idx in unique_labels:
            label_file_prefix = f"{image_file_prefix}_region_{int(idx):0>2d}"
            label_file = os.path.join(dataset_dir, "labels", label_file_prefix)
            label_file += ".npy"

            curr_label = (vol_label == idx).astype(np.float32)
            os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)
            np.save(label_file, curr_label)

            label_count += 1
            data_list.append(
                {
                    "image": image_file.replace(dataset_dir + os.pathsep, "") if relative_path else image_file,
                    "label": label_file.replace(dataset_dir + os.pathsep, "") if relative_path else label_file,
                    "region": int(idx),
                }
            )

    if unique_labels_count >= 20:
        logging.warning(f"Unique labels {unique_labels_count} exceeds 20. Please check if this is correct.")

    logging.info(
        "{} => Image Shape: {} => {}; Label Shape: {} => {}; Unique Labels: {}".format(
            vol_idx,
            vol_image.shape,
            image_count,
            vol_label.shape if vol_label is not None else None,
            label_count,
            unique_labels_count,
        )
    )
    return data_list
