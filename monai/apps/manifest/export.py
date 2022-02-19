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

import torch
from monai.apps import ConfigParser
from ignite.handlers import Checkpoint
from monai.data import save_net_with_metadata
from monai.networks import convert_to_torchscript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', '-w', type=str, help='file path of the trained model weights', required=True)
    parser.add_argument('--config', '-c', type=str, help='file path of config file that defines network', required=True)
    parser.add_argument('--meta', '-e', type=str, help='file path of the meta data')
    args = parser.parse_args()

    # load config file
    with open(args.config) as f:
        config_dict = json.load(f)
    # load meta data
    with open(args.meta) as f:
        meta_dict = json.load(f)

    net: torch.nn.Module = None
    # TODO: parse network definiftion from config file and construct network instance
    config_parser = ConfigParser(config_dict)
    net = config_parser.get_instance("network")

    checkpoint = torch.load(args.weights)
    # here we use ignite Checkpoint to support nested weights and be compatible with MONAI CheckpointSaver
    Checkpoint.load_objects(to_load={"model": net}, checkpoint=checkpoint)

    # convert to TorchScript model and save with meta data
    net = convert_to_torchscript(model=net)

    save_net_with_metadata(
        jit_obj=net,
        filename_prefix_or_stream="model.ts",
        include_config_vals=False,
        append_timestamp=False,
        meta_values=meta_dict,
        more_extra_files={args.config: json.dumps(config_dict).encode()},
    )


if __name__ == '__main__':
    main()
