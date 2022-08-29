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

import os
import sys

from ._version import get_versions

PY_REQUIRED_MAJOR = 3
PY_REQUIRED_MINOR = 7

version_dict = get_versions()
__version__: str = version_dict.get("version", "0+unknown")
__revision_id__: str = version_dict.get("full-revisionid")
del get_versions, version_dict

__copyright__ = "(c) MONAI Consortium"

__basedir__ = os.path.dirname(__file__)

if sys.version_info.major != PY_REQUIRED_MAJOR or sys.version_info.minor < PY_REQUIRED_MINOR:
    import warnings

    warnings.warn(
        f"MONAI requires Python {PY_REQUIRED_MAJOR}.{PY_REQUIRED_MINOR} or higher. "
        f"But the current Python is: {sys.version}",
        category=RuntimeWarning,
    )

from .utils.module import load_submodules  # noqa: E402

# handlers_* have some external decorators the users may not have installed
# *.so files and folder "_C" may not exist when the cpp extensions are not compiled
excludes = "(^(monai.handlers))|(^(monai.bundle))|(^(monai.fl))|((\\.so)$)|(^(monai._C))"

# load directory modules only, skip loading individual files
load_submodules(sys.modules[__name__], False, exclude_pattern=excludes)

# load all modules, this will trigger all export decorations
load_submodules(sys.modules[__name__], True, exclude_pattern=excludes)

__all__ = [
    "apps",
    "auto3dseg",
    "bundle",
    "config",
    "data",
    "engines",
    "fl",
    "handlers",
    "inferers",
    "losses",
    "metrics",
    "networks",
    "optimizers",
    "transforms",
    "utils",
    "visualize",
]
