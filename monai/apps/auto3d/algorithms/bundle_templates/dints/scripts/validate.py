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

import argparse
import json
import logging
import monai
import nibabel as nib
import numpy as np
import os
import pandas as pd
import pathlib
import shutil
import sys
import tempfile
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml

from datetime import datetime
from glob import glob
from torch import nn
from torch.nn.parallel import DistributedDataParallel

# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    AsDiscrete,
    # BatchInverseTransform,
    # Invertd,
)
from monai.data import (
    DataLoader,
    Dataset,
    create_test_image_3d,
    DistributedSampler,
    list_data_collate,
    partition_dataset,
)
# from monai.inferers import sliding_window_inference

# from monai.losses import DiceLoss, FocalLoss, GeneralizedDiceLoss
from monai.metrics import compute_meandice
from monai.utils import set_determinism
# from monai.utils.enums import InverseKeys
# from transforms import creating_transforms_testing, str2aug
# from utils import (
#     parse_monai_specs,
#     # parse_monai_transform_specs,
# )
from transforms import creating_transforms_offline_validation
# from utils import custom_compute_meandice


def run(config_file: Union[str, Sequence[str]], ckpt_path: str):
    # # disable logging for processes except 0 on every node
    # if args.local_rank != 0:
    #     f = open(os.devnull, "w")
    #     sys.stdout = sys.stderr = f

    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # if not os.path.exists(args.output_root):
    #     os.makedirs(args.output_root)

    # configuration
    with open(args.config) as in_file:
        config = yaml.full_load(in_file)
    # print("\n", pd.DataFrame(config), "\n")

    # core
    config_core = config["core"]
    input_channels = config_core["input_channels"]
    output_classes = config_core["output_classes"]

    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")

    # data
    with open(args.json, "r") as f:
        json_data = json.load(f)

    # inference data
    dataset_key = args.json_key
    files = []
    for _i in range(len(json_data[dataset_key])):
        # str_img = os.path.join(args.root, json_data[dataset_key][_i]["image"])
        str_seg = os.path.join(args.root, json_data[dataset_key][_i]["label"])
        str_pred = os.path.join(args.output_root, json_data[dataset_key][_i]["label"].split(os.sep)[-1])
        # str_pred = str_pred.replace("case_", "prediction_") + ".nii.gz"

        # if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
        # if not os.path.exists(str_img):
        #     continue
        if not os.path.exists(str_seg):
            continue

        # files.append({"image": str_img, "label": str_seg})
        # files.append({"image": str_img})
        files.append({"label": str_seg, "pred": str_pred})

    valid_files = files
    valid_files = partition_dataset(data=valid_files, shuffle=False, num_partitions=dist.get_world_size(), even_divisible=False)[dist.get_rank()]
    # print("valid_files", len(valid_files))

    # label_interpolation_transform = creating_label_interpolation_transform(label_interpolation, spacing, output_classes)
    # train_transforms = creating_transforms_training(foreground_crop_margin, label_interpolation_transform, num_patches_per_image, patch_size, scale_intensity_range, augmenations)
    # valid_transforms = creating_transforms_testing(foreground_crop_margin, scale_intensity_range, spacing)
    valid_transforms = creating_transforms_offline_validation(keys=["label", "pred"])

    argmax = AsDiscrete(
                 argmax=True,
                 to_onehot=False,
                 n_classes=output_classes
             )

    onehot = AsDiscrete(
                 argmax=False,
                 to_onehot=True,
                 n_classes=output_classes
             )

    # post_processing = KeepLargestConnectedComponent(
    #             applied_labels=[1, 2],
    #             independent=False,
    #             connectivity=None
    #         )

    # train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    # valid_ds = monai.data.CacheDataset(data=valid_files, transform=valid_transforms, cache_rate=1.0, num_workers=4)
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # valid_ds = monai.data.Dataset(data=valid_files, transform=valid_transforms)
    valid_ds = monai.data.Dataset(data=valid_files, transform=valid_transforms)

    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

    # network architecture
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    start_time = time.time()
    
    metric = torch.zeros((output_classes - 1) * 2, dtype=torch.float, device=device)
    metric_sum = 0.0
    metric_count = 0
    metric_mat = []

    _index = 0
    for valid_data in valid_loader:
        infer_labels = None
        infer_outputs = None

        infer_labels = valid_data["label"]
        infer_outputs = valid_data["pred"]

        infer_labels = onehot(infer_labels)
        infer_outputs = onehot(infer_outputs)

        value = compute_meandice(
            y_pred=infer_outputs, y=infer_labels, include_background=False
        )

        metric_count += len(value)
        metric_sum += value.sum().item()
        metric_vals = value.cpu().numpy()
        if len(metric_mat) == 0:
            metric_mat = metric_vals
        else:
            metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

        # print(_index + 1, "/", len(valid_loader), value)
        print_message = ""
        print_message += str(_index + 1)
        print_message += ", "
        print_message += valid_data["pred_meta_dict"]["filename_or_obj"][0].split(os.sep)[-1]
        print_message += ", "
        for _k in range(1, output_classes):
            if output_classes == 2:
                print_message += "{0:.5f}".format(metric_vals.squeeze())
            else:
                print_message += "{0:.5f}".format(metric_vals.squeeze()[_k - 1])
            print_message += ", "
        print(print_message)

        for _c in range(output_classes - 1):
            val0 = torch.nan_to_num(value[0, _c], nan=0.0)
            val1 = 1.0 - torch.isnan(value[0, 0]).float()
            metric[2 * _c] += val0 * val1
            metric[2 * _c + 1] += val1

        _index += 1

    # synchronizes all processes and reduce results
    dist.barrier()
    dist.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
    metric = metric.tolist()
    if dist.get_rank() == 0:
        for _c in range(output_classes - 1):
            print("evaluation metric - class {0:d}:".format(_c + 1), metric[2 * _c] / metric[2 * _c + 1])
        avg_metric = 0
        for _c in range(output_classes - 1):
            avg_metric += metric[2 * _c] / metric[2 * _c + 1]
        avg_metric = avg_metric / float(output_classes - 1)
        print("avg_metric", avg_metric)

        dict_file = {}
        dict_file["acc"] = float(avg_metric)
        for _c in range(output_classes - 1):
            dict_file["acc_class" + str(_c + 1)] = metric[2 * _c] / metric[2 * _c + 1]

        with open(os.path.join(args.output_root, "summary.yaml"), "w") as out_file:
            documents = yaml.dump(dict_file, stream=out_file)

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()