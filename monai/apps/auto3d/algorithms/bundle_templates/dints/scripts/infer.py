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

import json
import logging
import monai
import os
import sys
import torch

from monai import transforms
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, decollate_batch
from monai.inferers import sliding_window_inference
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

    ckpt_name = parser.get_parsed_content("infer")["ckpt_name"]
    data_list_key = parser.get_parsed_content("infer")["data_list_key"]
    output_path = parser.get_parsed_content("infer")["ouptut_path"]

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    infer_transforms = parser.get_parsed_content("transforms_infer")

    with open(data_list_file_path) as f:
        json_data = json.load(f)

    list_data = []
    for item in json_data[data_list_key]:
        list_data.append(item)

    files = []
    for _i in range(len(list_data)):
        str_img = os.path.join(data_file_base_dir, list_data[_i]["image"])

        if not os.path.exists(str_img):
            continue

        files.append({"image": str_img})

    infer_files = files

    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = ThreadDataLoader(infer_ds, num_workers=2, batch_size=1, shuffle=False)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = parser.get_parsed_content("network")
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    pretrained_ckpt = torch.load(ckpt_name, map_location=device)
    model.load_state_dict(pretrained_ckpt)
    print(f"[info] checkpoint {ckpt_name:s} loaded")

    post_transforms = [
        transforms.Invertd(
            keys="pred",
            transform=infer_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        )
    ]

    if softmax:
        post_transforms += [transforms.AsDiscreted(keys="pred", argmax=True)]
    else:
        post_transforms += [transforms.AsDiscreted(keys="pred", threshold=0.5)]

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

    model.eval()
    with torch.no_grad():
        for d in infer_loader:
            torch.cuda.empty_cache()

            infer_images = d["image"].to(device)

            with torch.cuda.amp.autocast():
                d["pred"] = sliding_window_inference(
                    infer_images,
                    patch_size_valid,
                    num_sw_batch_size,
                    model,
                    mode="gaussian",
                    overlap=overlap_ratio,
                )

            d = [post_transforms(i) for i in decollate_batch(d)]

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
