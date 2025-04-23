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

import json
import os
import pathlib
import random
from pathlib import Path
import numpy as np

import os
from monai.transforms import LoadImaged, ConcatItemsd, EnsureChannelFirstD, Compose, SaveImageD

from pathlib import Path
import numpy as np
import shutil
from monai.apps.nnunet import nnUNetV2Runner
from monai.bundle import ConfigParser
import monai

class NIFTINameFormatter:
    def __init__(self, suffix):
        self.suffix = suffix
    def __call__(self, metadict: dict, saver) -> dict:
        """Returns a kwargs dict for :py:meth:`FolderLayout.filename`,
        according to the input metadata and SaveImage transform."""
        subject = (
            metadict.get(monai.utils.ImageMetaKey.FILENAME_OR_OBJ, getattr(saver, "_data_index", 0))
            if metadict
            else getattr(saver, "_data_index", 0)
        )
        patch_index = metadict.get(monai.utils.ImageMetaKey.PATCH_INDEX, None) if metadict else None
        subject = subject[:-len(self.suffix)]+".nii.gz"
        #subject = subject[:-len(self.filename_key)]+".nii.gz"
        return {"subject": f"{subject}", "idx": patch_index}
    
def subfiles(folder, prefix=None, suffix=None, join=True, sort=True):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if prefix is not None:
        files = [f for f in files if f.startswith(prefix)]
    if suffix is not None:
        files = [f for f in files if f.endswith(suffix)]
    if sort:
        files.sort()
    if join:
        files = [os.path.join(folder, f) for f in files]
    return files

def concatenate_modalities(dataset_format,data_dir, modality_dict, output_data_dir=None, patient_id_in_file_identifier=True):

    if dataset_format == "decathlon" or dataset_format == "nnunet":
        imgs_output_dir = Path(output_data_dir).joinpath("imagesTr")
        labels_output_dir = Path(output_data_dir).joinpath("labelsTr")
        imgs_output_dir.mkdir(parents=True, exist_ok=True)
        labels_output_dir.mkdir(parents=True, exist_ok=True)

        keys = list(modality_dict.keys())
        keys.remove("label")
        transform = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstD(keys=keys),
        ConcatItemsd(keys=keys, name="image"),
        SaveImageD(keys=["image"], output_dir=imgs_output_dir, output_postfix="",separate_folder=False),
        ])

        label_files = subfiles(data_dir, prefix="labelsTr", suffix= modality_dict["label"], join=False)
        
        case_ids = [label_file[:-len(modality_dict["label"])] for label_file in label_files]
        for case_id in case_ids:
            data = {}
            for modality_id in modality_dict:
                if modality_id != "label":
                    data[modality_id] = str(Path(data_dir).joinpath("imagesTr").joinpath(case_id + modality_dict[modality_id])) 
            transform(data)
            shutil.copy(Path(data_dir).joinpath(f"labelsTr/{case_id}"+ modality_dict[modality_id]), Path(labels_output_dir).joinpath(f"{case_id}"+ modality_dict[modality_id]))
    elif dataset_format == "subfolders":
        keys = list(modality_dict.keys())
        keys.remove("label")

        formatter = NIFTINameFormatter(modality_dict[keys[0]])
        transform = Compose([
            LoadImaged(keys=keys),
            EnsureChannelFirstD(keys=keys),
            ConcatItemsd(keys=keys, name="image"),
            SaveImageD(keys=["image"], output_dir=data_dir, output_name_formatter=formatter, output_postfix="image",separate_folder=True),
        ])

        for patient_id in os.listdir(data_dir):
            data = {}
            for modality_id in modality_dict:
                if modality_id != "label":
                    if patient_id_in_file_identifier:
                        data[modality_id] = str(Path(data_dir).joinpath(patient_id, patient_id + modality_dict[modality_id]))
                    else:
                        data[modality_id] = str(Path(data_dir).joinpath(patient_id, modality_dict[modality_id]))
            transform(data)

def prepare_data_folder_api(data_dir,
    nnunet_root_dir,
    dataset_name_or_id,
    modality_dict,
    experiment_name,
    dataset_format,
    modality_list = None,
    subfolder_suffix=None,
    patient_id_in_file_identifier=True,
    trainer_class_name="nnUNetTrainer",
    concatenate_modalities = False,
    output_data_dir=None,
    ):
    if concatenate_modalities:
        concatenate_modalities(dataset_format,data_dir, modality_dict, output_data_dir=output_data_dir, patient_id_in_file_identifier=patient_id_in_file_identifier)
        if dataset_format == "decathlon" or dataset_format == "nnunet":
            modality_dict = {"image": modality_dict[list(modality_dict.keys())[0]], "label": modality_dict["label"]}
        elif dataset_format == "subfolders":
            modality_dict = {"image": "_image.nii.gz", "label": modality_dict["label"]}

    if dataset_format == "subfolders":
        if subfolder_suffix is not None:
            data_list = {
                "training": [
                    {
                        modality_id: (
                            str(
                                pathlib.Path(f.name).joinpath(
                                    f.name[: -len(subfolder_suffix)] + modality_dict[modality_id]
                                )
                            )
                            if patient_id_in_file_identifier
                            else str(pathlib.Path(f.name).joinpath(modality_dict[modality_id]))
                        )
                        for modality_id in modality_dict
                    }
                    for f in sorted(os.scandir(data_dir), key=lambda e: e.name)
                    if f.is_dir()
                ],
                "testing": [],
            }
        else:
            data_list = {
                "training": [
                    {
                        modality_id: (
                            str(pathlib.Path(f.name).joinpath(f.name + modality_dict[modality_id]))
                            if patient_id_in_file_identifier
                            else str(pathlib.Path(f.name).joinpath(modality_dict[modality_id]))
                        )
                        for modality_id in modality_dict
                    }
                    for f in sorted(os.scandir(data_dir), key=lambda e: e.name)
                    if f.is_dir()
                ],
                "testing": [],
            }
    elif dataset_format == "decathlon" or dataset_format == "nnunet":
        cases = []

        for f in sorted(os.scandir(Path(data_dir).joinpath("imagesTr")), key=lambda e: e.name):
            if f.is_file():
                for modality_suffix in list(modality_dict.values()):
                    if f.name.endswith(modality_suffix) and modality_suffix != ".nii.gz":
                        cases.append(f.name[: -len(modality_suffix)])
                if len(np.unique(list(modality_dict.values()))) == 1 and ".nii.gz" in list(modality_dict.values()):
                    cases.append(f.name[: -len(".nii.gz")])
        cases = np.unique(cases)
        data_list = {
            "training": [
                {
                    modality_id: str(Path("imagesTr").joinpath(case + modality_dict[modality_id]))
                    for modality_id in modality_dict
                    if modality_id != "label"
                }
                for case in cases
            ],
            "testing": [],
        }
        for idx, case in enumerate(data_list["training"]):
            modality_id = list(modality_dict.keys())[0]
            case_id = Path(case[modality_id]).name[: -len(modality_dict[modality_id])]
            data_list["training"][idx]["label"] = str(Path("labelsTr").joinpath(case_id + modality_dict["label"]))
    else:
        raise ValueError("Dataset format not supported")

    for idx, train_case in enumerate(data_list["training"]):
        for modality_id in modality_dict:
            data_list["training"][idx][modality_id + "_is_file"] = (
                Path(data_dir).joinpath(data_list["training"][idx][modality_id]).is_file()
            )
            if "image" not in data_list["training"][idx] and modality_id != "label":
                data_list["training"][idx]["image"] = data_list["training"][idx][modality_id]
        data_list["training"][idx]["fold"] = 0

    random.seed(42)
    random.shuffle(data_list["training"])

    data_list["testing"] = [data_list["training"][0]]

    num_folds = 5
    fold_size = len(data_list["training"]) // num_folds
    for i in range(num_folds):
        for j in range(fold_size):
            data_list["training"][i * fold_size + j]["fold"] = i

    datalist_file = Path(data_dir).joinpath(f"{experiment_name}_folds.json")
    with open(datalist_file, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    os.makedirs(nnunet_root_dir, exist_ok=True)

    if modality_list is None:
        modality_list = [k for k in modality_dict.keys() if k != "label"]
    
    data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")
    data_src = {
        "modality": modality_list,
        "dataset_name_or_id": dataset_name_or_id,
        "datalist": str(datalist_file),
        "dataroot": str(data_dir),
    }

    ConfigParser.export_config_file(data_src, data_src_cfg)

    if dataset_format != "nnunet":
        runner = nnUNetV2Runner(
            input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir
        )
        runner.convert_dataset()
    else:
        ...

    return data_list