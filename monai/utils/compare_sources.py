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
This module is used in Github Actions to determine if a Python source file has been modified in a PR.
"""

from __future__ import annotations

import ast
import sys
from os.path import splitext

from monai.config.type_definitions import PathLike

SKIP_EXTS = (".md", ".rst")


class RemoveDocstrings(ast.NodeTransformer):
    """
    Strips docstrings from source files so that they aren't used for comparing ASTs.
    """

    def visit(self, node: ast.AST) -> ast.AST:
        try:
            # remove docstrings from the files
            if ast.get_docstring(node) is not None:
                del node.body[0]
        except TypeError:
            pass
        return super().visit(node)


def sources_equal(src1: str, src2: str) -> bool:
    """
    Compare Python source texts at the AST level without docstrings or comments. If two texts are equal except for
    changes to docstrings or comments, they will be considered equal by this function. Any other changes will appear
    as differences in the AST representations, which are compared inefficiently by comparing tree string dumps.

    Args:
        src1: first Python source text.
        src2: second Python source text.

    Returns:
        True if the texts `src1` and `src2` are equal with docstrings and comments removed, False otherwise.
    """
    remdoc = RemoveDocstrings()

    m1: ast.Module = remdoc.generic_visit(ast.parse(src1))
    m2: ast.Module = remdoc.generic_visit(ast.parse(src2))

    list1 = list(ast.walk(m1))
    list2 = list(ast.walk(m2))

    if len(list1) != len(list2) or any(type(n1) is not type(n2) for n1, n2 in zip(list1, list2)):
        return False

    return ast.dump(m1) == ast.dump(m2)


def files_considered_equal(file1: PathLike, file2: PathLike) -> bool:
    """
    Returns True if the files are considered equal, that is they are doc files or differ only in docstrings or comments.

    Args:
        file1: first file path to compare.
        file2: second file path to compare.

    Returns:
        True if the files have the same extension, and either this extension is in `SKIP_EXTS`, or is ".py" with the
        files having the same content according to `sources_equal`. If the extensions differ, or are both ".py" but the
        file contents differ, returns False.
    """
    _, ext1 = splitext(str(file1))
    _, ext2 = splitext(str(file2))

    # if extensions aren't equal then definitely different (which shouldn't happen anyway)
    if ext1 != ext2:
        return False

    # if extensions are an ignored type, ie. docs, don't compare at all
    if ext1 in SKIP_EXTS:
        return True

    # if not Python source files, they aren't doc files at this point so assume different
    if ext1 != ".py":
        return False

    # compare the actual parsed contents of the source files
    with open(file1) as o1, open(file2) as o2:
        return sources_equal(o1.read(), o2.read())


if __name__ == "__main__":
    _, file1, file2 = sys.argv
    sys.exit(0 if files_considered_equal(file1, file2) else 1)
