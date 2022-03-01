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
import logging
import sys

from monai.bundle.utils import parse_config_files, parse_id_value


def run():
    """
    Specify a metadata file and a config file to run a regular training or evaluation program.
    It's used to execute most of the supervised training, evaluation or inference cases.
    It supports to override the config content with specified `id` and `value` pairs.
    The `override` arg can also be used to provide default value for placeholders. For example:
    put a placeholder `"data": "@runtime_value"` in the config, then define `runtime_value` in `override`.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", "-m", type=str, help="filepath of the metadata file.", required=True)
    parser.add_argument("--config", "-c", type=str, help="filepath of the config file.", required=True)
    parser.add_argument("--override", "-o", metavar="ID=VALUE", nargs="*")
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help=("ID name of the target workflow, it must have the `run` method, follow MONAI `BaseWorkflow`."),
        required=True,
    )

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    override = {}
    if args.override is not None:
        for pair in args.override:
            id, v = parse_id_value(pair)
            override[id] = v
    config_parser = parse_config_files(config_file=args.config, meta_file=args.metadata, override=override)

    workflow = config_parser.get_parsed_content(id=args.target)
    workflow.run()


if __name__ == "__main__":
    run()
