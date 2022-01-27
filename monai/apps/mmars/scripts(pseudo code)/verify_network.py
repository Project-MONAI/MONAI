
# Copyright 2020 - 2021 MONAI Consortium
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
from monai.utils.type_conversion import get_equivalent_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='config file that defines components', required=True)
    parser.add_argument('--meta', '-e', type=str, help='file path of the meta data')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = {}

    # load meta data
    with open(args.meta, "r") as f:
        configs.update(json.load(f))
    # load config file, can override meta data in config
    with open(args.config, "r") as f:
        configs.update(json.load(f))

    model: torch.nn.Module = None
    # TODO: parse inference config file and construct instances
    config_parser = ConfigParser(configs)

    model = config_parser.get_instance("model")
    input_channels = config_parser.get_config("network_data_format#inputs#image#num_channels")
    input_spatial_shape = tuple(config_parser.get_config("network_data_format#inputs#image#spatial_shape"))
    dtype = config_parser.get_config("network_data_format#inputs#image#dtype")
    dtype = get_equivalent_dtype(dtype, data_type=torch.Tensor)

    output_channels = config_parser.get_config("network_data_format#outputs#pred#num_channels")
    output_spatial_shape = tuple(config_parser.get_config("network_data_format#outputs#pred#spatial_shape"))

    model.eval()
    with torch.no_grad():
        test_data = torch.rand(*(input_channels, *input_spatial_shape), dtype=dtype, device=device)
        output = model(test_data)
        if output.shape[0] != output_channels:
            raise ValueError(f"channel number of output data doesn't match expection: {output_channels}.")
        if output.shape[1:] != output_spatial_shape:
            raise ValueError(f"spatial shape of output data doesn't match expection: {output_spatial_shape}.")


if __name__ == '__main__':
    main()
