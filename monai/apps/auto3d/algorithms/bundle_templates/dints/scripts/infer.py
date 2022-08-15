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

from typing import Sequence, Union
import glob

import torch
from monai.bundle import ConfigParser
from monai.data import decollate_batch


def run(config_file: Union[str, Sequence[str]], ckpt_path: str):
    parser = ConfigParser()
    parser.read_config(config_file)
    # edit the config content at runtime for input / output information and lazy instantiation
    datalist = list(sorted(glob.glob("/workspace/data/Task09_Spleen/imagesTs/*.nii.gz")))
    input_data = [{f"{parser['image_key']}": i} for i in datalist]
    output_dir = "/workspace/data/tutorials/modules/bundle/hybrid_programming/eval"
    parser["input_data"] = input_data
    parser["output_dir"] = output_dir
    parser["inferer"]["roi_size"] = [160, 160, 160]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # instantialize the components
    model = parser.get_parsed_content("network").to(device)
    model.load_state_dict(torch.load(ckpt_path))

    dataloader = parser.get_parsed_content("dataloader")
    if len(dataloader) == 0:
        raise ValueError("no data found in the dataloader, please ensure the input image paths are accessable.")
    inferer = parser.get_parsed_content("inferer")
    postprocessing = parser.get_parsed_content("postprocessing")

    model.eval()
    with torch.no_grad():
        for d in dataloader:
            images = d[parser["image_key"]].to(device)
            # define sliding window size and batch size for windows inference
            d[parser["pred_key"]] = inferer(inputs=images, network=model)
            # decollate the batch data into a list of dictionaries, then execute postprocessing transforms
            [postprocessing(i) for i in decollate_batch(d)]


if __name__ == "__main__":
    from monai.utils import optional_import

    fire, _ = optional_import("fire")
    fire.Fire()
