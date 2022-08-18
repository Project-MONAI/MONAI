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

import csv
import json
import logging
import monai
import numpy as np
import os
import sys
import torch
import yaml

from monai import transforms
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from typing import Sequence, Union


def run(
    config_file: Optional[Union[str, Sequence[str]]] = None,
    **override,
):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    _args = _update_args(config_file=config_file, **override)
    config_file_ = _pop_args(_args, "config_file")[0]

    parser = ConfigParser()
    parser.read_config(config_file_)
    parser.update(pairs=_args)

    data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
    data_list_file_path = parser.get_parsed_content("data_list_file_path")
    fold = parser.get_parsed_content("fold")
    input_channels = parser.get_parsed_content("input_channels")
    num_sw_batch_size = parser.get_parsed_content("num_sw_batch_size")
    output_classes = parser.get_parsed_content("output_classes")
    overlap_ratio = parser.get_parsed_content("overlap_ratio")
    patch_size_valid = parser.get_parsed_content("patch_size_valid")
    softmax = parser.get_parsed_content("softmax")

    ckpt_name = parser.get_parsed_content("validate")["ckpt_name"]
    output_path = parser.get_parsed_content("validate")["ouptut_path"]
    save_mask = parser.get_parsed_content("validate")["save_mask"]

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    infer_transforms = parser.get_parsed_content("transforms_infer")
    validate_transforms = transforms.Compose(
        [
            infer_transforms,
            transforms.LoadImaged(keys="label"),
            transforms.EnsureChannelFirstd(keys="label"),
            transforms.EnsureTyped(keys="label"),
        ]
    )

    with open(data_list_file_path) as f:
        json_data = json.load(f)

    list_valid = []
    for item in json_data["training"]:
        if item["fold"] == fold:
            item.pop("fold", None)
            list_valid.append(item)

    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(data_file_base_dir, list_valid[_i]["image"])
        str_seg = os.path.join(data_file_base_dir, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    val_files = files

    val_ds = monai.data.Dataset(data=val_files, transform=validate_transforms)
    val_loader = ThreadDataLoader(val_ds, num_workers=2, batch_size=1, shuffle=False)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = parser.get_parsed_content("network")
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    pretrained_ckpt = torch.load(ckpt_name, map_location=device)
    model.load_state_dict(pretrained_ckpt)
    print(f"[info] checkpoint {ckpt_name:s} loaded")

    if softmax:
        post_pred = transforms.Compose(
            [
                transforms.EnsureType(),
                transforms.AsDiscrete(to_onehot=output_classes),
            ]
        )
    else:
        post_pred = transforms.Compose([EnsureType()])

    post_transforms = [
        transforms.Invertd(
            keys="pred",
            transform=validate_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
    ]

    if softmax:
        post_transforms += [transforms.AsDiscreted(keys="pred", argmax=True)]
    else:
        post_transforms += [transforms.AsDiscreted(keys="pred", threshold=0.5)]

    if save_mask:
        post_transforms += [
            transforms.SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=output_path,
                output_postfix="seg",
                resample=False,
            )
        ]

    post_transforms = transforms.Compose(post_transforms)

    metric_dim = output_classes - 1 if softmax else output_classes
    metric_sum = 0.0
    metric_count = 0
    metric_mat = []

    row = ["case_name"]
    for _i in range(metric_dim):
        row.append("class_" + str(_i + 1))

    with open(os.path.join(output_path, "raw.csv"), "w", encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    model.eval()
    with torch.no_grad():
        metric = torch.zeros(metric_dim * 2, dtype=torch.float)

        _index = 0
        for d in val_loader:
            torch.cuda.empty_cache()

            val_images = d["image"].to(device)
            val_labels = d["label"]

            with torch.cuda.amp.autocast():
                d["pred"] = sliding_window_inference(
                    val_images,
                    patch_size_valid,
                    num_sw_batch_size,
                    model,
                    mode="gaussian",
                    overlap=overlap_ratio,
                )

            d = [post_transforms(i) for i in decollate_batch(d)]

            val_outputs = post_pred(d[0]["pred"])
            val_outputs = val_outputs[None, ...]

            if softmax:
                val_labels = post_pred(val_labels[0, ...])
                val_labels = val_labels[None, ...]

            value = compute_meandice(
                y_pred=val_outputs, y=val_labels, include_background=not softmax
            )

            metric_count += len(value)
            metric_sum += value.sum().item()
            metric_vals = value.cpu().numpy()
            if len(metric_mat) == 0:
                metric_mat = metric_vals
            else:
                metric_mat = np.concatenate((metric_mat, metric_vals), axis=0)

            print_message = ""
            print_message += str(_index + 1)
            print_message += ", "
            print_message += d[0]["pred"].meta["filename_or_obj"]
            print_message += ", "
            for _k in range(metric_dim):
                if output_classes == 2:
                    print_message += f"{metric_vals.squeeze():.5f}"
                else:
                    print_message += f"{metric_vals.squeeze()[_k]:.5f}"
                print_message += ", "
            print(print_message)

            row = [d[0]["pred"].meta["filename_or_obj"]]
            for _i in range(metric_dim):
                row.append(metric_vals[0, _i])

            with open(os.path.join(output_path, "raw.csv"), "a", encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            for _c in range(metric_dim):
                val0 = torch.nan_to_num(value[0, _c], nan=0.0)
                val1 = 1.0 - torch.isnan(value[0, 0]).float()
                metric[2 * _c] += val0 * val1
                metric[2 * _c + 1] += val1

            _index += 1

        metric = metric.tolist()
        for _c in range(metric_dim):
            print(f"evaluation metric - class {_c + 1:d}:", metric[2 * _c] / metric[2 * _c + 1])
        avg_metric = 0
        for _c in range(metric_dim):
            avg_metric += metric[2 * _c] / metric[2 * _c + 1]
        avg_metric = avg_metric / float(metric_dim)
        print("avg_metric", avg_metric)

        dict_file = {}
        dict_file["acc"] = float(avg_metric)
        for _c in range(metric_dim):
            dict_file["acc_class" + str(_c + 1)] = metric[2 * _c] / metric[2 * _c + 1]

        with open(os.path.join(output_path, "summary.yaml"), "w") as out_file:
            documents = yaml.dump(dict_file, stream=out_file)

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
