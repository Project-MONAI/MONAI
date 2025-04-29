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

import SimpleITK as sitk
import subprocess
import yaml
import os
from monai.transforms import LoadImaged, ConcatItemsd, EnsureChannelFirstD, Compose, SaveImageD

from pathlib import Path
import numpy as np
import shutil
from monai.apps.nnunet import nnUNetV2Runner
from monai.bundle import ConfigParser
import monai
from monai.data import load_decathlon_datalist
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance, DiceMetric
from monai.transforms import AsDiscrete
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
            print(f"Processing case: {case_id}")
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
            print(f"Processing case: {case_id}")
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
    concatenate_modalities_flag = False,
    output_data_dir=None,
    labels = None,
    regions_class_order = None,
    ):
    if concatenate_modalities_flag:
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
    if labels is not None:
        data_src["labels"] = labels
    if regions_class_order is not None:
        data_src["regions_class_order"] = regions_class_order

    ConfigParser.export_config_file(data_src, data_src_cfg)

    if dataset_format != "nnunet":
        runner = nnUNetV2Runner(
            input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir
        )
        runner.convert_dataset()
    else:
        ...

    return data_list


def cross_site_evaluation_api(nnunet_root_dir, dataset_name_or_id, app_path, app_model_path, app_output_path, fold=0, trainer_class_name="nnUNetTrainer", nnunet_plans_name="nnUNetPlans", skip_prediction=False):
    data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")

    runner = nnUNetV2Runner(input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir)

    with open(Path(nnunet_root_dir).joinpath(f"Task{dataset_name_or_id}_data_src_cfg.yaml"), "r") as f:
        nnunet_config = yaml.safe_load(f)

    data_root_dir = nnunet_config["dataroot"]
    data_list_file = nnunet_config["datalist"]
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    data_list = load_decathlon_datalist(data_list_file_path=data_list_file, base_dir=data_root_dir)

    dataset_name = maybe_convert_to_dataset_name(int(dataset_name_or_id))

    with open(Path(runner.nnunet_raw).joinpath(dataset_name, "datalist.json"), "r") as f:
        nnunet_datalist = yaml.safe_load(f)
        
    with open(Path(runner.nnunet_preprocessed).joinpath(dataset_name, "splits_final.json"), "r") as f:
        nnunet_splits = json.load(f)

    id_mapping = {}

    for data in data_list:
        app_input_path = data["image"]
        filename = Path(app_input_path).name
        
        new_id = None  
        for case in nnunet_datalist["training"]:
            if case["image"].endswith(filename):
                new_id = case["new_name"]
                break
        if new_id in nnunet_splits[fold]["val"]:
            id_mapping[Path(data["image"]).name.split("_")[0].split(".")[0]] = new_id
            if skip_prediction:
                continue
            subprocess.run(
                [
                "python",
                app_path,
                "--input", data["image"]
                ],
                env={
                    **os.environ,
                    "HOLOSCAN_MODEL_PATH": app_model_path,
                    "HOLOSCAN_OUTPUT_PATH": app_output_path,
                }
            )

    for file in os.listdir(app_output_path):
        if file.endswith(".nii.gz"):
            if skip_prediction:
                continue
            id = id_mapping[file[:-len(".nii.gz")]]
            shutil.move(
                os.path.join(app_output_path, file),
                os.path.join(app_output_path, f"{id}.nii.gz")
            )     

    dataset_file = os.path.join(
            runner.nnunet_results,
            runner.dataset_name,
            f"{trainer_class_name}__{nnunet_plans_name}__3d_fullres",
            "dataset.json",
        )

    with open(dataset_file, "r") as f:
        dataset_dict = json.load(f)
        labels = dataset_dict["labels"]
        labels = {str(v): k for k, v in labels.items()}
            
    validation_summary_dict = compute_validation_metrics(str(Path(runner.nnunet_raw).joinpath(dataset_name, "labelsTr")), app_output_path, n_labels=len(labels)-1)
    
    return validation_summary_dict, labels

def compute_validation_metrics(gt_folder, pred_folder, n_labels=1):
    print("Computing validation metrics...")
    print("Number of labels: ", n_labels)
    to_onehot = AsDiscrete(to_onehot=n_labels+1)
    dice_fn = DiceMetric(include_background=False)
    summary_file = Path(pred_folder).joinpath("summary.json")
    
    if not summary_file.exists():
        summary = {'metric_per_case': []}
        for file in os.listdir(pred_folder):
            if file.endswith(".nii.gz"):
                summary['metric_per_case'].append(
                    {
                        "prediction_file": os.path.join(pred_folder, file),
                        "metrics": {
                        "1": {}
                        }
                    }
                )
    else:
        with open(summary_file) as f:
            summary = json.load(f)
    for idx, case in enumerate(summary['metric_per_case']):
        print("Processing case: ", idx)

        case_id = Path(case['prediction_file']).name[:-len(".nii.gz")]
        gt_image = sitk.ReadImage(Path(gt_folder).joinpath(case_id+".nii.gz"))
        pred_image = sitk.ReadImage(Path(pred_folder).joinpath(case_id+".nii.gz"))
        gt_array = sitk.GetArrayFromImage(gt_image)
        pred_array = sitk.GetArrayFromImage(pred_image)

        hd_95 = compute_hausdorff_distance(
        to_onehot(pred_array[None])[None],
        to_onehot(gt_array[None])[None],
        spacing = gt_image.GetSpacing(),
        percentile=95
        )

        asd = compute_average_surface_distance(
        to_onehot(pred_array[None])[None],
        to_onehot(gt_array[None])[None],
        spacing = gt_image.GetSpacing(),
        )
        dice = dice_fn(to_onehot(pred_array[None])[None], to_onehot(gt_array[None])[None])
        for label_id in range(1,1+n_labels):
            summary['metric_per_case'][idx]["metrics"][str(label_id)]["HD95"] = hd_95[label_id-1][0].item()
            summary['metric_per_case'][idx]["metrics"][str(label_id)]["ASD"] = asd[label_id-1][0].item()
            summary['metric_per_case'][idx]["metrics"][str(label_id)]["Dice"] = dice[label_id-1][0].item()

    for label_id in range(1,1+n_labels):
        summary["mean"] = {}
        summary["mean"][str(label_id)] = {}
        summary["mean"][str(label_id)]["HD95"] = np.mean(
            [case["metrics"][str(label_id)]["HD95"] for case in summary['metric_per_case'] if not np.isnan(case["metrics"][str(label_id)]["HD95"]) and not np.isinf(case["metrics"][str(label_id)]["HD95"])]
        )
        summary["mean"][str(label_id)]["ASD"] = np.mean([case["metrics"][str(label_id)]["ASD"] for case in summary['metric_per_case'] if not np.isnan(case["metrics"][str(label_id)]["ASD"]) and not np.isinf(case["metrics"][str(label_id)]["ASD"])])
        summary["mean"][str(label_id)]["Dice"] = np.mean([case["metrics"][str(label_id)]["Dice"] for case in summary['metric_per_case']  if not np.isnan(case["metrics"][str(label_id)]["Dice"]) and not np.isinf(case["metrics"][str(label_id)]["Dice"])])
    
    return summary


def plan_and_preprocess_api(nnunet_root_dir, dataset_name_or_id, trainer_class_name="nnUNetTrainer", nnunet_plans_name="nnUNetPlans"):
    data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")

    runner = nnUNetV2Runner(input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir)

    runner.plan_and_process(
        npfp=2, verify_dataset_integrity=True, c=["3d_fullres"], n_proc=[2], overwrite_plans_name=nnunet_plans_name
    )

    preprocessed_folder = runner.nnunet_preprocessed

    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    dataset_name = maybe_convert_to_dataset_name(int(dataset_name_or_id))

    with open(Path(preprocessed_folder).joinpath(f"{dataset_name}", nnunet_plans_name + ".json"), "r") as f:
        nnunet_plans = json.load(f)
        
    return nnunet_plans