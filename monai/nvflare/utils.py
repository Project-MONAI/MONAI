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

import torch
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
from monai.apps.nnunet.nnunet_bundle import convert_monai_bundle_to_nnunet

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
            print(f"Processing case: {patient_id}")
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
    elif dataset_format == "monai-label":
        data_list = data_dir
    else:
        raise ValueError("Dataset format not supported")

    for idx, train_case in enumerate(data_list["training"]):
        for modality_id in modality_dict:
            if dataset_format == "monai-label":
                data_list["training"][idx][modality_id + "_is_file"] = (
                    Path(data_list["training"][idx][modality_id]).is_file()
                )
            else:
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

    if dataset_format == "monai-label":
        monai_label_data_dir = Path(data_list["training"][0]["image"]).parent
        datalist_file = Path(monai_label_data_dir).joinpath(f"{experiment_name}_folds.json")
        dataroot = str(monai_label_data_dir)
    else:
        datalist_file = Path(data_dir).joinpath(f"{experiment_name}_folds.json")
        dataroot = str(data_dir)
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
        "dataroot": dataroot,
    }
    if labels is not None:
        print("Labels: ", labels)
        data_src["labels"] = labels
        for label in labels:
            if isinstance(labels[label], str):
                labels[label] = labels[label].split(",")
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


def cross_site_evaluation_api(nnunet_root_dir, dataset_name_or_id, app_path, app_model_path, app_output_path, fold=0, trainer_class_name="nnUNetTrainer", nnunet_plans_name="nnUNetPlans", skip_prediction=False, original_path=None):
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
        updated_image_path = False
        for case in nnunet_datalist["training"]:
            if case["image"].endswith(filename):
                new_id = case["new_name"]
                break
            if filename.startswith(case["new_name"]+"_"):
                new_id = case["new_name"]
                data["image"] = os.path.join(original_path, Path(case["image"]).name)
                app_input_path = data["image"]
                updated_image_path = True
                break
        if new_id in nnunet_splits[fold]["val"]:
            if updated_image_path:
                id_mapping[Path(data["image"]).name.split(".")[0]] = new_id
            else:
                id_mapping[Path(data["image"]).name.split("_")[0].split(".")[0]] = new_id
            if skip_prediction:
                continue
            print(f"Processing case: {new_id}")
            print(f"App input path: {app_input_path}")
            mapped_filename = Path(data["image"]).name.split(".")[0]
            print(f"Mapping: {mapped_filename} -> {new_id}")
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
            summary['metric_per_case'][idx]["metrics"][str(label_id)] = {}
            summary['metric_per_case'][idx]["metrics"][str(label_id)]["HD95"] = hd_95[0][label_id-1].item()
            summary['metric_per_case'][idx]["metrics"][str(label_id)]["ASD"] = asd[0][label_id-1].item()
            summary['metric_per_case'][idx]["metrics"][str(label_id)]["Dice"] = dice[0][label_id-1].item()

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

def prepare_bundle_api(bundle_config, train_extra_configs=None, is_federated=False):
    """
    Prepare and update MONAI bundle configuration files for training and evaluation, supporting both standard and federated workflows.
    This function loads, modifies, and saves configuration files (YAML/JSON) for MONAI nnUNet bundles, injecting runtime parameters,
    handling federated learning specifics, and updating label dictionaries and metrics. It also manages MLflow tracking parameters
    and ensures all necessary configuration files are present and up-to-date.
    Parameters
    ----------
    bundle_config : dict
        Dictionary containing bundle configuration parameters. Expected keys:
            - bundle_root (str): Root directory of the bundle.
            - tracking_uri (str): URI for MLflow tracking.
            - mlflow_experiment_name (str): MLflow experiment name.
            - mlflow_run_name (str): MLflow run name.
            - dataset_name_or_id (str): Dataset identifier or name.
            - label_dict (dict): Mapping of label indices to label names.
            - nnunet_plans_identifier (str, optional): Identifier for nnUNet plans.
            - nnunet_trainer_class_name (str, optional): Name of the nnUNet trainer class.
            - dataset_name (str, optional): Human-readable dataset name.
    train_extra_configs : dict, optional
        Additional training configuration parameters. May include:
            - resume_epoch (int): Epoch to resume training from.
            - region_based (bool): Whether to use region-based postprocessing and metrics.
        Any other keys will be injected into the training configuration.
    is_federated : bool, default=False
        Whether to prepare the configuration for federated learning.
    Returns
    -------
    dict
        Dictionary containing the updated configuration objects:
            - "evaluate_config": The evaluation configuration dictionary.
            - "train_config": The training configuration dictionary.
    Notes
    -----
    - This function modifies and overwrites configuration files in-place within the bundle directory.
    - Handles both standard and federated learning scenarios, including metric renaming and handler removal for federated mode.
    - Updates label dictionaries and ensures consistency across all configuration files.
    """
    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "train.yaml")) as f:
        train_config = yaml.safe_load(f)
        train_config["bundle_root"] = bundle_config["bundle_root"]
        train_config["tracking_uri"] = bundle_config["tracking_uri"]
        train_config["mlflow_experiment_name"] = bundle_config["mlflow_experiment_name"]
        train_config["mlflow_run_name"] = bundle_config["mlflow_run_name"]

        train_config["dataset_name_or_id"] = bundle_config["dataset_name_or_id"]
        train_config["data_src_cfg"] = "$@nnunet_root_folder+'/Task'+@dataset_name_or_id+'_data_src_cfg.yaml'"
        train_config["nnunet_root_folder"] = "."
        train_config["runner"] = {
            "_target_": "nnUNetV2Runner",
            "input_config": "$@data_src_cfg",
            "trainer_class_name": "@nnunet_trainer_class_name",
            "work_dir": "@nnunet_root_folder",
        }


        if is_federated:
            train_config["network"] = "$@nnunet_trainer.network._orig_mod"

            train_handlers = train_config["train_handlers"]["handlers"]
            for idx, handler in enumerate(train_handlers):
                if handler["_target_"] == "ValidationHandler":
                    train_handlers.pop(idx)
                    break

            train_config["train_handlers"]["handlers"] = train_handlers

        if train_extra_configs is not None and "resume_epoch" in train_extra_configs:
            resume_epoch = train_extra_configs["resume_epoch"]
            train_config["initialize"] = [
                "$monai.utils.set_determinism(seed=123)",
                "$@runner.dataset_name_or_id",
                f"$src.trainer.reload_checkpoint(@train#trainer, {resume_epoch}, @iterations, @ckpt_dir, @lr_scheduler)",
            ]
        else:
            train_config["initialize"] = ["$monai.utils.set_determinism(seed=123)", "$@runner.dataset_name_or_id"]
            
        if train_extra_configs is not None:
            for key in train_extra_configs:
                if key != "resume_epoch":
                    train_config[key] = train_extra_configs[key]

        if is_federated:
            if "Val_Dice" in train_config["val_key_metric"]:
                train_config["val_key_metric"] = {"Val_Dice_Local": train_config["val_key_metric"]["Val_Dice"]}

            if "Val_Dice_per_class" in train_config["val_additional_metrics"]:
                train_config["val_additional_metrics"] = {
                    "Val_Dice_per_class_Local": train_config["val_additional_metrics"]["Val_Dice_per_class"]
                }
        if "nnunet_plans_identifier" in bundle_config:
            train_config["nnunet_plans_identifier"] = bundle_config["nnunet_plans_identifier"]

        if "nnunet_trainer_class_name" in bundle_config:
            train_config["nnunet_trainer_class_name"] = bundle_config["nnunet_trainer_class_name"]


    train_config["label_dict"] = {
        "0": "background",
    }
    
    for k, v in bundle_config["label_dict"].items():
        if k != "0":
            train_config["label_dict"][str(v)] = k

    if "nnUNet_n_proc_DA" in os.environ and int(os.environ["nnUNet_n_proc_DA"]) == 0:
        train_config["train"]["train_data"] = "$[{'case_identifier':k} for k in @nnunet_trainer.dataloader_train.data_loader._data.identifiers]" 
    else:
        train_config["train"]["train_data"] = "$[{'case_identifier':k} for k in @nnunet_trainer.dataloader_train.generator._data.identifiers]"
        
    if train_extra_configs is not None and "region_based" in train_extra_configs:
        if "train_postprocessing_label_based" not in train_config:
            train_config["train_postprocessing_label_based"] = train_config["train_postprocessing"]
            train_config["train_postprocessing"] = train_config["train_postprocessing_region_based"]
        if is_federated:
            train_config["val_additional_metrics"]["Val_Dice_per_class_Local"]["include_background"] = True
            train_config["val_key_metric"]["Val_Dice_Local"]["include_background"] = True
        else:
            train_config["val_additional_metrics"]["Val_Dice_per_class"]["include_background"] = True
            train_config["val_key_metric"]["Val_Dice"]["include_background"] = True
        train_config["train_additional_metrics"]["Train_Dice_per_class"]["include_background"] = True
        train_config["train_key_metric"]["Train_Dice"]["include_background"] = True

    
    train_config["num_classes"] = len(train_config["label_dict"])
    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "train.json"), "w") as f:
        json.dump(train_config, f)

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "train.yaml"), "w") as f:
        yaml.dump(train_config, f)

    if not Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.yaml").exists():
        shutil.copy(
            Path(bundle_config["bundle_root"]).joinpath("nnUNet", "evaluator", "evaluator.yaml"),
            Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.yaml"),
        )

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.yaml")) as f:
        evaluate_config = yaml.safe_load(f)
        evaluate_config["bundle_root"] = bundle_config["bundle_root"]

        evaluate_config["tracking_uri"] = bundle_config["tracking_uri"]
        evaluate_config["mlflow_experiment_name"] = bundle_config["mlflow_experiment_name"]
        evaluate_config["mlflow_run_name"] = bundle_config["mlflow_run_name"]

        if "nnunet_plans_identifier" in bundle_config:
            evaluate_config["nnunet_plans_identifier"] = bundle_config["nnunet_plans_identifier"]
        if "nnunet_trainer_class_name" in bundle_config:
            evaluate_config["nnunet_trainer_class_name"] = bundle_config["nnunet_trainer_class_name"]

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.json"), "w") as f:
        json.dump(evaluate_config, f)

    with open(Path(bundle_config["bundle_root"]).joinpath("configs", "evaluate.yaml"), "w") as f:
        yaml.dump(evaluate_config, f)

    with open(Path(bundle_config["bundle_root"]).joinpath("nnUNet", "params.yaml")) as f:
        mlflow_params = yaml.safe_load(f)

        mlflow_params["dataset_name_or_id"] = bundle_config["dataset_name_or_id"]
        mlflow_params["experiment_name"] = bundle_config["mlflow_experiment_name"]
        mlflow_params["mlflow_run_name"] = bundle_config["mlflow_run_name"]
        mlflow_params["tracking_uri"] = bundle_config["tracking_uri"]
        mlflow_params["num_classes"] = len(train_config["label_dict"])
        mlflow_params["label_dict"] = train_config["label_dict"]

        if "nnunet_plans_identifier" in bundle_config:
            mlflow_params["nnunet_plans_identifier"] = bundle_config["nnunet_plans_identifier"]

        if "nnunet_trainer_class_name" in bundle_config:
            mlflow_params["nnunet_trainer_class_name"] = bundle_config["nnunet_trainer_class_name"]
        
        if "dataset_name_or_id" in bundle_config:
            mlflow_params["dataset_name_or_id"] = bundle_config["dataset_name_or_id"]

    with open(Path(bundle_config["bundle_root"]).joinpath("nnUNet", "params.yaml"), "w") as f:
        yaml.dump(mlflow_params, f)
    
    return {"evaluate_config": evaluate_config, "train_config": train_config}

def train_api(nnunet_root_dir, dataset_name_or_id, experiment_name, trainer_class_name="nnUNetTrainer", run_with_bundle=False, bundle_root=None, skip_training=False, continue_training=False, fold=0, tracking_uri=None, client_name=None, resume_epoch=None):
    
    data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")
    runner = nnUNetV2Runner(input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir)

    if not skip_training:
        if not run_with_bundle:
            if continue_training:
                runner.train_single_model(config="3d_fullres", fold=fold, c="")
            else:
                runner.train_single_model(config="3d_fullres", fold=fold)
        else:
            os.environ["BUNDLE_ROOT"] = bundle_root
            os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + bundle_root
            config_files = os.path.join(bundle_root, "configs", "train.yaml")
            if continue_training:
                config_files = [os.path.join(bundle_root, "configs", "train.yaml"), os.path.join(bundle_root, "configs", "train_resume.yaml")]
            monai.bundle.run(
                config_file=config_files,
                bundle_root=bundle_root,
                nnunet_trainer_class_name=trainer_class_name,
                mlflow_experiment_name=experiment_name,
                mlflow_run_name="run_" + client_name,
                tracking_uri=tracking_uri,
                fold_id=fold,
                nnunet_root_folder=nnunet_root_dir,
                reload_checkpoint_epoch=resume_epoch
            )
    nnunet_config = {"dataset_name_or_id": dataset_name_or_id, "nnunet_trainer": trainer_class_name}
    convert_monai_bundle_to_nnunet(nnunet_config, bundle_root)
    runner.train_single_model(config="3d_fullres", fold=fold, val="")


def validation_api(nnunet_root_dir, dataset_name_or_id, trainer_class_name="nnUNetTrainer", nnunet_plans_name="nnUNetPlans", fold=0):
    data_src_cfg = os.path.join(nnunet_root_dir, f"Task{dataset_name_or_id}_data_src_cfg.yaml")
    runner = nnUNetV2Runner(input_config=data_src_cfg, trainer_class_name=trainer_class_name, work_dir=nnunet_root_dir)
    runner.train_single_model(config="3d_fullres", fold=fold, val="")
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

    validation_summary_dict = compute_validation_metrics(str(Path(runner.nnunet_raw).joinpath(runner.dataset_name, "labelsTr")),
                                   str(Path(runner.nnunet_results).joinpath(runner.dataset_name,f"{trainer_class_name}__{nnunet_plans_name}__3d_fullres", f"fold_{fold}", "validation")),
                                   len(labels)-1)
    
    return validation_summary_dict, labels

def finalize_bundle_api(nnunet_root_dir, bundle_root, trainer_class_name="nnUNetTrainer", fold=0, is_federated=False):
    if nnunet_root_dir is None:
        raise ValueError("nnunet_root_dir must be provided if validate_with_nnunet is True")
    if not Path(bundle_root).joinpath("models", "plans.json").exists():
        raise ValueError("plans.json file not found in the models directory of the bundle")
    if not Path(bundle_root).joinpath("models", "dataset.json").exists():
        raise ValueError("dataset.json file not found in the models directory of the bundle")
    
    print("Converting bundle to nnUNet format...")
    
    with open(Path(bundle_root).joinpath("models","plans.json"),"r") as f:
        plans = json.load(f)

    with open(Path(bundle_root).joinpath("configs","plans.yaml"),"w") as f:
        yaml.dump({"plans": plans}, f)

    with open(Path(bundle_root).joinpath("models","dataset.json"),"r") as f:
        dataset_json = json.load(f)
        
    with open(Path(bundle_root).joinpath("configs","dataset.yaml"),"w") as f:
        yaml.dump({"dataset_json": dataset_json}, f)
    
    if is_federated:
        checkpoint = {
            "trainer_name": trainer_class_name,
            "inference_allowed_mirroring_axes": (0, 1, 2),
            "init_args": {
                "configuration": "3d_fullres",
        }
        }
        
        torch.save(checkpoint, Path(bundle_root).joinpath("models","nnunet_checkpoint.pth"))
    
        checkpoint_dict = torch.load(Path(bundle_root).joinpath("models",f"fold_{fold}","FL_global_model.pt"))
        
        new_checkpoint_dict = {}
        new_checkpoint_dict["network_weights"] = checkpoint_dict["model"]
        torch.save(new_checkpoint_dict, Path(bundle_root).joinpath("models",f"fold_{fold}","checkpoint_epoch=1000.pt"))
        torch.save(new_checkpoint_dict, Path(bundle_root).joinpath("models",f"fold_{fold}","checkpoint_key_metric=1.0.pt"))