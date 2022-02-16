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

from typing import Dict, List, Union


def is_instantiable(config: Union[Dict, List, str]) -> bool:
    """
    Check whether the content of the config represents a `class` or `function` to instantiate
    with specified "<path>" or "<name>".

    Args:
        config: input config content to check.

    """
    return isinstance(config, dict) and ("<path>" in config or "<name>" in config)


def is_expression(config: Union[Dict, List, str]) -> bool:
    """
    Check whether the content of the config is executable expression string.
    If True, the string should start with "$" mark.

    Args:
        config: input config content to check.

    """
    return isinstance(config, str) and config.startswith("$")
