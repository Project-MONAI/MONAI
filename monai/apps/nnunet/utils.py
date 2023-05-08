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

import copy
import os

import numpy as np

import monai
from monai.bundle import ConfigParser
from monai.utils import StrEnum, ensure_tuple, optional_import

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")
nib, _ = optional_import("nibabel")

logger = monai.apps.utils.get_logger(__name__)

__all__ = ["analyze_data", "create_new_data_copy", "create_new_dataset_json", "NNUNETMode"]


class NNUNETMode(StrEnum):
    N_2D = "2d"
    N_3D_FULLRES = "3d_fullres"
    N_3D_LOWRES = "3d_lowres"
    N_3D_CASCADE_FULLRES = "3d_cascade_fullres"


def analyze_data(datalist_json: dict, data_dir: str) -> tuple[int, int]:
    """
    Analyze (training) data

    Args:
        datalist_json: original data list .json (required by most monai tutorials).
        data_dir: raw data directory.
    """
    img = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
        os.path.join(data_dir, datalist_json["training"][0]["image"])
    )
    num_input_channels = img.size()[0] if img.dim() == 4 else 1
    logger.info(f"num_input_channels: {num_input_channels}")

    num_foreground_classes = 0
    for _i in range(len(datalist_json["training"])):
        seg = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
            os.path.join(data_dir, datalist_json["training"][_i]["label"])
        )
        num_foreground_classes = max(num_foreground_classes, int(seg.max()))
    logger.info(f"num_foreground_classes: {num_foreground_classes}")

    return num_input_channels, num_foreground_classes


def create_new_data_copy(
    test_key: str, datalist_json: dict, data_dir: str, num_input_channels: int, output_datafolder: str
) -> None:
    """
    Create and organize a new copy of data to meet the requirements of nnU-Net V2

    Args:
        test_key: key for test data in the data list .json.
        datalist_json: original data list .json (required by most monai tutorials).
        data_dir: raw data directory.
        num_input_channels: number of input (image) channels.
        output_datafolder: output folder.
    """
    _index = 0
    new_datalist_json: dict = {"training": [], test_key: []}

    for _key, _folder, _label_folder in list(
        zip(["training", test_key], ["imagesTr", "imagesTs"], ["labelsTr", "labelsTs"])
    ):
        if _key is None:
            continue

        logger.info(f"converting data section: {_key}...")
        for _k in tqdm(range(len(datalist_json[_key]))) if has_tqdm else range(len(datalist_json[_key])):
            orig_img_name = (
                datalist_json[_key][_k]["image"]
                if isinstance(datalist_json[_key][_k], dict)
                else datalist_json[_key][_k]
            )
            img_name = f"case_{_index}"
            _index += 1

            # copy image
            nda = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
                os.path.join(data_dir, orig_img_name)
            )
            affine = nda.meta["original_affine"]
            nda = nda.numpy()
            for _l in range(num_input_channels):
                outimg = nib.Nifti1Image(nda[_l, ...], affine)
                index = "_" + str(_l + 10000)[-4:]
                nib.save(outimg, os.path.join(output_datafolder, _folder, img_name + index + ".nii.gz"))

            # copy label
            if isinstance(datalist_json[_key][_k], dict) and "label" in datalist_json[_key][_k]:
                nda = monai.transforms.LoadImage(image_only=True, ensure_channel_first=True, simple_keys=True)(
                    os.path.join(data_dir, datalist_json[_key][_k]["label"])
                )
                affine = nda.meta["original_affine"]
                nda = nda.numpy().astype(np.uint8)
                nda = nda[0, ...] if nda.ndim == 4 and nda.shape[0] == 1 else nda
                nib.save(
                    nib.Nifti1Image(nda, affine), os.path.join(output_datafolder, _label_folder, img_name + ".nii.gz")
                )

            if isinstance(datalist_json[_key][_k], dict):
                _val = copy.deepcopy(datalist_json[_key][_k])
                _val["new_name"] = img_name
                new_datalist_json[_key].append(_val)
            else:
                new_datalist_json[_key].append({"image": datalist_json[_key][_k], "new_name": img_name})

            ConfigParser.export_config_file(
                config=new_datalist_json,
                filepath=os.path.join(output_datafolder, "datalist.json"),
                fmt="json",
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )

    return


def create_new_dataset_json(
    modality: str, num_foreground_classes: int, num_input_channels: int, num_training_data: int, output_filepath: str
) -> None:
    """
    Create a new copy of dataset .json to meet the requirements of nnU-Net V2

    Args:
        modality: image modality, could a string or a list of strings.
        num_foreground_classes: number of foreground classes.
        num_input_channels: number of input (image) channels.
        num_training_data: number of training data.
        output_filepath: output file path/name.
    """
    new_json_data: dict = {}

    # modality = self.input_info.pop("modality")
    modality = ensure_tuple(modality)  # type: ignore

    new_json_data["channel_names"] = {}
    for _j in range(num_input_channels):
        new_json_data["channel_names"][str(_j)] = modality[_j]

    new_json_data["labels"] = {}
    new_json_data["labels"]["background"] = 0
    for _j in range(num_foreground_classes):
        new_json_data["labels"][f"class{_j + 1}"] = _j + 1

    # new_json_data["numTraining"] = len(datalist_json["training"])
    new_json_data["numTraining"] = num_training_data
    new_json_data["file_ending"] = ".nii.gz"

    ConfigParser.export_config_file(
        config=new_json_data,
        # filepath=os.path.join(raw_data_foldername, "dataset.json"),
        filepath=output_filepath,
        fmt="json",
        sort_keys=True,
        indent=4,
        ensure_ascii=False,
    )

    return
