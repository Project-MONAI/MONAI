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

from monai.apps.manifest.utils import verify_metadata


def verify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", "-m", type=str, help="filepath of the metadata file.", required=True)
    parser.add_argument("--schema_url", "-u", type=str, help="filepath of the config file.", required=True)
    parser.add_argument("--filepath", "-f", type=str, help="filepath to store downloaded schema.", required=True)
    parser.add_argument("--hash_val", "-v", type=str, help="MD5 hash value to verify schema file.", required=False)

    args = parser.parse_args()
    verify_metadata(
        metadata=args.metadata,
        schema_url=args.schema_url,
        filepath=args.filepath,
        create_dir=True,
        hash_val=args.hash_val,
    )


if __name__ == "__main__":
    verify()
