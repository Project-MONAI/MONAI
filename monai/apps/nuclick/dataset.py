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

import copy
import json
import logging
import os
import pathlib
from typing import Dict, List

import numpy as np

from monai.apps.utils import tqdm
from monai.utils import optional_import

loadmat, _ = optional_import("scipy.io", name="loadmat")
PILImage, _ = optional_import("PIL.Image")


def consep_nuclei_dataset(datalist, output_dir, crop_size, min_area=80, min_distance=20, limit=0) -> List[Dict]:
    """
    Utility to pre-process and create dataset list for Patches per Nuclei for training over ConSeP dataset.

    Args:
        datalist: A list of data dictionary. Each entry should at least contain 'image_key': <image filename>.
            For example, typical input data can be a list of dictionaries::

                [{'image': <image filename>, 'label': <label filename>}]

        output_dir: target directory to store the training data after flattening
        crop_size: Crop Size for each patch
        min_area: Min Area for each nuclei to be included in dataset
        min_distance: Min Distance from boundary for each nuclei to be included in dataset
        limit: limit number of inputs for pre-processing.  Defaults to 0 (no limit).

    Raises:
        ValueError: When ``datalist`` is Empty
        ValueError: When ``scipy.io.loadmat`` is Not available

    Returns:
        A new datalist that contains path to the images/labels after pre-processing.

    Example::

        datalist = consep_nuclei_dataset(
            datalist=[{'image': 'img1.png', 'label': 'label1.mat'}],
            output_dir=output,
            crop_size=128,
            limit=1,
        )

        print(datalist[0]["image"], datalist[0]["label"])
    """

    if not len(datalist):
        raise ValueError("Input datalist is empty")

    if not loadmat:
        print("Please make sure scipy with loadmat function is correctly installed")
        raise ValueError("scipy.io.loadmat module/function not found")

    dataset_json: List[Dict] = []
    for d in tqdm(datalist):
        logging.debug(f"Processing Image: {d['image']} => Label: {d['label']}")

        # Image
        image = PILImage.open(d["image"]).convert("RGB")

        # Label
        m = loadmat(d["label"])
        instances = m["inst_map"]

        for nuclei_id, (class_id, (y, x)) in enumerate(zip(m["inst_type"], m["inst_centroid"]), start=1):
            x, y = (int(x), int(y))
            class_id = int(class_id)
            class_id = 3 if class_id in (3, 4) else 4 if class_id in (5, 6, 7) else class_id  # override

            if 0 < limit <= len(dataset_json):
                return dataset_json

            item = __prepare_patch(
                d=d,
                nuclei_id=nuclei_id,
                output_dir=output_dir,
                image=image,
                instances=instances,
                instance_idx=nuclei_id,
                crop_size=crop_size,
                class_id=class_id,
                centroid=(x, y),
                min_area=min_area,
                min_distance=min_distance,
                others_idx=255,
            )

            if item:
                dataset_json.append(item)

    return dataset_json


def __prepare_patch(
    d,
    nuclei_id,
    output_dir,
    image,
    instances,
    instance_idx,
    crop_size,
    class_id,
    centroid,
    min_area,
    min_distance,
    others_idx=255,
):
    image_np = np.array(image)
    image_size = image.size

    bbox = __compute_bbox(crop_size, centroid, image_size)

    cropped_label_np = instances[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    cropped_label_np = np.array(cropped_label_np)

    this_label = np.where(cropped_label_np == instance_idx, class_id, 0)
    if np.count_nonzero(this_label) < min_area:
        return None

    x, y = centroid
    if x < min_distance or y < min_distance or (image_size[0] - x) < min_distance or (image_size[1] - y < min_distance):
        return None

    centroid = centroid[0] - bbox[0], centroid[1] - bbox[1]
    others = np.where(np.logical_and(cropped_label_np > 0, cropped_label_np != instance_idx), others_idx, 0)
    cropped_label_np = this_label + others
    cropped_label = PILImage.fromarray(cropped_label_np.astype(np.uint8), None)

    cropped_image_np = image_np[bbox[0] : bbox[2], bbox[1] : bbox[3], :]
    cropped_image = PILImage.fromarray(cropped_image_np, "RGB")

    images_dir = os.path.join(output_dir, "Images") if output_dir else "Images"
    labels_dir = os.path.join(output_dir, "Labels") if output_dir else "Labels"
    centroids_dir = os.path.join(output_dir, "Centroids") if output_dir else "Centroids"

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(centroids_dir, exist_ok=True)

    image_id = pathlib.Path(d["image"]).stem
    file_prefix = f"{image_id}_{class_id}_{str(instance_idx).zfill(4)}"
    image_file = os.path.join(images_dir, f"{file_prefix}.png")
    label_file = os.path.join(labels_dir, f"{file_prefix}.png")
    centroid_file = os.path.join(centroids_dir, f"{file_prefix}.txt")

    cropped_image.save(image_file)
    cropped_label.save(label_file)
    with open(centroid_file, "w") as fp:
        json.dump([centroid], fp)

    item = copy.deepcopy(d)
    item["nuclei_id"] = nuclei_id
    item["mask_value"] = class_id
    item["image"] = image_file
    item["label"] = label_file
    item["centroid"] = centroid
    return item


def __compute_bbox(patch_size, centroid, size):
    x, y = centroid
    m, n = size

    x_start = int(max(x - patch_size / 2, 0))
    y_start = int(max(y - patch_size / 2, 0))
    x_end = x_start + patch_size
    y_end = y_start + patch_size
    if x_end > m:
        x_end = m
        x_start = m - patch_size
    if y_end > n:
        y_end = n
        y_start = n - patch_size
    return x_start, y_start, x_end, y_end
