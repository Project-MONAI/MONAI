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

import re
import sys
from collections.abc import Collection

BUILD_SYSTEM_KEY = "build-system"
PROJ_KEY = "project"
OPTS_KEY = "optional-dependencies"
DEP_KEY = "dependencies"
NAME_KEY = "name"
REQ_KEY = "requires"
TOML_FILE = "pyproject.toml"


# a requirement of the form "<name>[<extra>,...]" and nothing else; a version specifier or an
# environment marker after the brackets makes it something other than a bare self-reference
SELF_REF_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*\[([^\]]+)\]$")


def _normalize_name(name: str) -> str:
    """
    Normalize a project or extra name so case and ``-``/``_``/``.`` spellings compare equal.

    PEP 503 (project names) and PEP 685 (extra names) specify the same rule, so one function
    serves both sides of the comparison.
    """
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _expand_self_extras(dependencies: list[str], name: str, opts: dict) -> list[str]:
    """
    Replace self-referential requirements such as ``monai[lint]`` with the contents of that group.

    pip resolves such a requirement against the package index, so leaving one in a generated
    requirements file installs the published release instead of the checkout being worked on.

    Args:
        dependencies: requirement strings, some of which may be self-references.
        name: this project's name, the only one treated as a self-reference.
        opts: the "optional-dependencies" table the groups are read from.

    Returns:
        List of requirements with every self-reference replaced by the group it names.

    Raises:
        KeyError: If a self-reference names a group absent from `opts`.
    """
    self_name = _normalize_name(name)
    groups = {_normalize_name(key): value for key, value in opts.items()}
    expanded: list[str] = []
    pending = list(dependencies)
    seen: set[str] = set()

    while pending:
        req = pending.pop(0)
        match = SELF_REF_RE.match(req.strip())
        if match is None or _normalize_name(match.group(1)) != self_name:
            expanded.append(req)
            continue
        for group in (_normalize_name(g) for g in match.group(2).split(",")):
            if group in seen:  # a group already expanded, or a cycle
                continue
            seen.add(group)
            pending.extend(groups[group])

    return expanded


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

    dependencies = _expand_self_extras(dependencies, proj[NAME_KEY], opts)

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
