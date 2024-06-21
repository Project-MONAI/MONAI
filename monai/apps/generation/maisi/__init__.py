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

import subprocess
import sys


def install_and_import(package, package_fullname=None):
    if package_fullname is None:
        package_fullname = package

    try:
        __import__(package)
    except ImportError:
        print(f"'{package}' is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_fullname])
        print(f"'{package}' installation completed.")
        __import__(package)


install_and_import("generative", "monai-generative")
