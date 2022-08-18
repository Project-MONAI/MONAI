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
import os
import sys
from typing import Dict, Optional, Sequence, Union

import torch

import monai
from monai import transforms
from monai.bundle import ConfigParser
from monai.bundle.scripts import _pop_args, _update_args
from monai.data import ThreadDataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference


class InferClass:
    def __init__(
        self, config_file: Optional[Union[str, Sequence[str]]] = None, **override
    ):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        _args = _update_args(config_file=config_file, **override)
        config_file_ = _pop_args(_args, "config_file")[0]

        parser = ConfigParser()
        parser.read_config(config_file)
        parser.update(pairs=_args)

        data_file_base_dir = parser.get_parsed_content("data_file_base_dir")
        data_list_file_path = parser.get_parsed_content("data_list_file_path")
        self.fast = parser.get_parsed_content("infer")["fast"]
        self.num_sw_batch_size = parser.get_parsed_content("num_sw_batch_size")
        self.overlap_ratio = parser.get_parsed_content("overlap_ratio")
        self.patch_size_valid = parser.get_parsed_content("patch_size_valid")
        softmax = parser.get_parsed_content("softmax")

        ckpt_name = parser.get_parsed_content("infer")["ckpt_name"]
        data_list_key = parser.get_parsed_content("infer")["data_list_key"]
        output_path = parser.get_parsed_content("infer")["ouptut_path"]

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        self.infer_transforms = parser.get_parsed_content("transforms_infer")

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

        self.infer_files = files

        self.infer_loader = None
        if self.fast:
            infer_ds = monai.data.Dataset(
                data=self.infer_files, transform=self.infer_transforms
            )
            self.infer_loader = ThreadDataLoader(
                infer_ds, num_workers=8, batch_size=1, shuffle=False
            )

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.model = parser.get_parsed_content("network")
        self.model = self.model.to(self.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        pretrained_ckpt = torch.load(ckpt_name, map_location=self.device)
        self.model.load_state_dict(pretrained_ckpt)
        print(f"[info] checkpoint {ckpt_name:s} loaded")

        post_transforms = [
            transforms.Invertd(
                keys="pred",
                transform=self.infer_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            transforms.Activationsd(
                keys="pred", softmax=softmax, sigmoid=not softmax
            ),
            transforms.CopyItemsd(
                keys="pred", times=1, names="pred_final"
            ),
        ]

        if softmax:
            post_transforms += [transforms.AsDiscreted(keys="pred_final", argmax=True)]
        else:
            post_transforms += [transforms.AsDiscreted(keys="pred_final", threshold=0.5)]

        post_transforms += [
            transforms.SaveImaged(
                keys="pred_final",
                meta_keys="pred_meta_dict",
                output_dir=output_path,
                output_postfix="seg",
                resample=False,
            )
        ]
        self.post_transforms = transforms.Compose(post_transforms)

        return

    @torch.no_grad()
    def infer(self, image_file):
        self.model.eval()

        batch_data = self.infer_transforms(image_file)
        batch_data = list_data_collate([batch_data])
        infer_image = batch_data["image"].to(self.device)

        with torch.cuda.amp.autocast():
            batch_data["pred"] = sliding_window_inference(
                infer_image,
                self.patch_size_valid,
                self.num_sw_batch_size,
                self.model,
                mode="gaussian",
                overlap=self.overlap_ratio,
            )

        batch_data = [self.post_transforms(i) for i in decollate_batch(batch_data)]

        return batch_data[0]["pred"]

    def infer_all(self):
        for _i in range(len(self.infer_files)):
            infer_filename = self.infer_files[_i]
            pred = self.infer(infer_filename)

        return

    @torch.no_grad()
    def batch_infer(self):
        self.model.eval()
        with torch.no_grad():
            for d in self.infer_loader:
                torch.cuda.empty_cache()

                infer_images = d["image"].to(self.device)

                with torch.cuda.amp.autocast():
                    d["pred"] = sliding_window_inference(
                        infer_images,
                        self.patch_size_valid,
                        self.num_sw_batch_size,
                        self.model,
                        mode="gaussian",
                        overlap=self.overlap_ratio,
                    )

                d = [self.post_transforms(i) for i in decollate_batch(d)]

        return


def run(config_file: Optional[Union[str, Sequence[str]]] = None, **override):
    infer_instance = InferClass(config_file, **override)
    if infer_instance.fast:
        print("[info] fast mode")
        infer_instance.batch_infer()
    else:
        print("[info] slow mode")
        infer_instance.infer_all()

    return


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
