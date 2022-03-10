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

from typing import Dict, Optional, Union

from monai.bundle.config_reader import ConfigReader

__all__ = ["ID_REF_KEY", "ID_SEP_KEY", "EXPR_KEY", "MACRO_KEY", "update_default_args"]


ID_REF_KEY = "@"  # start of a reference to a ConfigItem
ID_SEP_KEY = "#"  # separator for the ID of a ConfigItem
EXPR_KEY = "$"  # start of an ConfigExpression
MACRO_KEY = "%"  # start of a macro of a config


def update_default_args(args: Optional[Union[str, Dict]] = None, **kwargs) -> Dict:
    """
    Update the `args` with the input `kwargs`.
    For dict data, recursively update the content based on the keys.

    Args:
        args: source args to update.
        kwargs: destination args to update.

    """
    args_: Dict = args if isinstance(args, dict) else {}  # type: ignore
    if isinstance(args, str):
        # args are defined in a structured file
        args_ = ConfigReader.load_config_file(args)

    # recursively update the default args with new args
    for k, v in kwargs.items():
        args_[k] = update_default_args(args_[k], **v) if isinstance(v, dict) and isinstance(args_.get(k), dict) else v
    return args_
