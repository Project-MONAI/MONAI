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
from monai.apps.manifest.utils import parse_config_files, parse_id_value


def run():
    """
    Specify the metadata file, config file to run a standard training or evaluation program.
    It's used to execute most of the supervised training or evaluation cases.
    It supports to override the config content with specified `id` and `value`.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", "-m", type=str, help="filepath of the metadata file.", required=True)
    parser.add_argument("--config", "-c", type=str, help="filepath of the config file.", required=True)
    parser.add_argument("--override", "-o", metavar="ID=VALUE", nargs="*")
    parser.add_argument(
        "--target", "-c", type=str,
        help=("ID name of the target workflow, it must have the `run` method, follow MONAI `BaseWorkflow`."),
        required=True,
    )

    args = parser.parse_args()
    override = {}
    for pair in args.override:
        id, v = parse_id_value(pair)
        override[id] = v
    config_parser = parse_config_files(config_file=args.config, meta_file=args.metadata, override=override)

    workflow = config_parser.get_resolved_content(id=args.target)
    workflow.run()


if __name__ == '__main__':
    run()
