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
from pathlib import Path
from typing import Dict, Optional, Union

from monai.apps.utils import download_url
from monai.config import PathLike
from monai.utils import optional_import

validate, _ = optional_import("jsonschema", name="validate")
ValidationError, _ = optional_import("jsonschema.exceptions", name="ValidationError")


def verify_metadata(
    metadata: Union[Dict, str],
    schema_url: str,
    filepath: PathLike,
    create_dir: bool = True,
    hash_val: Optional[str] = None,
    result_path: Optional[PathLike] = None,
    **kwargs,
):
    """
    For more details: https://python-jsonschema.readthedocs.io/en/stable/validate/#jsonschema.validate.
    """
    def _check_dir(path: Path):
        path_dir = path.parent
        if not path_dir.exists():
            if create_dir:
                path_dir.mkdir(parents=True)
            else:
                raise ValueError(f"the directory of specified path is not existing: {path_dir}.")

    filepath = Path(filepath)
    _check_dir(path=filepath)
    download_url(url=schema_url, filepath=filepath, hash_val=hash_val, hash_type="md5", progress=True)

    # FIXME: will update to use `load_config_file()` when PR 3832 is merged
    with open(filepath) as f:
        schema = json.load(f)
    if isinstance(metadata, str):
        with open(metadata) as f:
            metadata = json.load(f)

    try:
        validate(instance=metadata, schema=schema, **kwargs)
    except ValidationError as e:
        if result_path is not None:
            result_path = Path(result_path)
            _check_dir(result_path)
            with open(result_path, "w") as f:
                f.write(str(e))
        raise e
