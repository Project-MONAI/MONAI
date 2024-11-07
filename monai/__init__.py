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

from __future__ import annotations

import logging
import os
import sys
import warnings

from ._version import get_versions

old_showwarning = warnings.showwarning


def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    ignore_files = ["ignite/handlers/checkpoint", "modelopt/torch/quantization/tensor_quant"]
    if any(ignore in filename for ignore in ignore_files):
        return
    old_showwarning(message, category, filename, lineno, file, line)


class DeprecatedTypesWarningFilter(logging.Filter):
    def filter(self, record):
        message_bodies_to_ignore = [
            "np.bool8",
            "np.object0",
            "np.int0",
            "np.uint0",
            "np.void0",
            "np.str0",
            "np.bytes0",
            "@validator",
            "@root_validator",
            "class-based `config`",
            "pkg_resources",
            "Implicitly cleaning up",
        ]
        for message in message_bodies_to_ignore:
            if message in record.getMessage():
                return False
        return True


# workaround for https://github.com/Project-MONAI/MONAI/issues/8060
# TODO: remove this workaround after upstream fixed the warning
# Set the custom warning handler to filter warning
warnings.showwarning = custom_warning_handler
# Get the logger for warnings and add the filter to the logger
logging.getLogger("py.warnings").addFilter(DeprecatedTypesWarningFilter())


PY_REQUIRED_MAJOR = 3
PY_REQUIRED_MINOR = 9

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
excludes = "|".join(
    [
        "(^(monai.handlers))",
        "(^(monai.bundle))",
        "(^(monai.fl))",
        "((\\.so)$)",
        "(^(monai._C))",
        "(.*(__main__)$)",
        "(.*(video_dataset)$)",
        "(.*(nnunet).*$)",
    ]
)

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

try:
    from .utils.tf32 import detect_default_tf32

    detect_default_tf32()
    import torch

    # workaround related to https://github.com/Project-MONAI/MONAI/issues/7575
    if hasattr(torch.cuda.device_count, "cache_clear"):
        torch.cuda.device_count.cache_clear()
except BaseException:
    from .utils.misc import MONAIEnvVars

    if MONAIEnvVars.debug():
        raise
