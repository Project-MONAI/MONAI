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

"""
This program prints the MONAI dependencies for the optional names given on the command line. The printed values can
be piped to a requirements file to work with pip. All required dependencies are always printed, those for builing are
included in "build-system" is given as an argument, and all optional requirements are included if "*" is given. This
assumes the pyproject.toml file is in the current working directory.
"""

from __future__ import annotations

import sys
from collections.abc import Collection

BUILD_SYSTEM_KEY = "build-system"
PROJ_KEY = "project"
OPTS_KEY = "optional-dependencies"
DEP_KEY = "dependencies"
REQ_KEY = "requires"
TOML_FILE = "pyproject.toml"


def parse_dependencies(filename: str | None = None, sections: Collection[str] | None = None) -> list[str]:
    """
    Parse the toml file given by `filename` and return the dependency sections selected by `sections`.

    Args:
        filename: TOML file to parse, if None this defaults to TOML_FILE.
        sections: "optional-dependencies" sections to print in addition to the required dependencies. If
            "build-system" is included, the build requirements will be included in the output. If "*" is included, all
            of the optional dependencies will be included in the output.

    Returns:
        List of requirements in alphabetical order.
    """
    # these imports should be here to avoid attempting to import when MONAI is imported and both packages are missing
    # isort: off
    if sys.version_info.minor >= 11:
        from tomllib import loads
    else:
        from tomli import loads
    # isort: on

    with open(filename or TOML_FILE) as o:
        data = loads(o.read())

    proj = data[PROJ_KEY]
    opts = proj[OPTS_KEY]
    dependencies = list(proj[DEP_KEY])
    sections = set(sections or [])

    if BUILD_SYSTEM_KEY in sections:
        sections.remove(BUILD_SYSTEM_KEY)
        dependencies += data[BUILD_SYSTEM_KEY][REQ_KEY]

    if "*" in sections:
        dependencies += sum(opts.values(), [])
    else:
        for s in sections:
            dependencies += opts[s]

    return sorted(set(dependencies))


def print_dependencies_argv():
    """
    Print dependencies specified through argv.
    """
    dependencies = parse_dependencies(sections=set(sys.argv[1:]))

    for d in dependencies:
        print(d)


if __name__ == "__main__":
    print_dependencies_argv()
