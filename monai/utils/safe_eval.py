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

import ast
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["SAFE_TYPES", "safe_eval"]

# default set of safe AST node types
SAFE_TYPES = (
    ast.Expression,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


def safe_eval(
    expr: str,
    globals_vars: Mapping[str, Any] | None = None,
    locals_vars: Mapping[str, object] | None = None,
    allowed_types: Sequence[type] = SAFE_TYPES,
):
    """
    Evaluate the Python expression `expr` using `eval`, but only if it is a safe expression in that its parsed AST
    contains nodes whose types are given in `allowed_types`. This ensures unsafe node types are excluded, if these
    are present in the AST a ValueError is raised. The default set of such types in `SAFE_TYPES` ensures only
    expressions with constants and names can be evaluated, so excludes attribute access, indexing, and calls. Code
    injection is infeasible through such expressions, so this is a safe and secure way of evaluating simple expressions.

    Args:
        expr: expression to evaluate, this will be stripped before parsing to avoid indentation complaints
        globals: global variable mapping
        locals: local variable mapping
        allowed_types: sequence of allowed AST types which can be found in `expr` when parsed

    Raises:
        ValueError: raised when any node in the AST parsed from `expr` has a type not in `allowed_types`

    Returns:
        The evaluated expression value, using `eval` with `globals` and `locals`
    """
    parsed = ast.parse(expr.strip(), mode="eval")

    def _disallowed_node(n):
        return not any(isinstance(n, at) for at in allowed_types)

    disallowed = list(filter(_disallowed_node, ast.walk(parsed)))

    if disallowed:
        disallowed_strs = list(map(ast.unparse, disallowed))
        raise ValueError(
            f"Unsafe expression `{expr}` cannot be evaluated, contains disallowed components: {disallowed_strs}"
        )

    return eval(expr, globals_vars, locals_vars)
