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

import numpy as np

__all__ = ["SAFE_TYPES", "safe_eval"]

# default set of safe AST node types
SAFE_TYPES: Sequence[type] = (
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


class _RewriteConstNp(ast.NodeTransformer):
    """Replaces int and float constants in the tree with those wrapped in Numpy types."""

    def __init__(self, int_type_str: str, float_type_str: str):
        self.int_type_str = int_type_str
        self.float_type_str = float_type_str

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return node
        type_str = self.int_type_str if isinstance(node.value, int) else self.float_type_str
        func_node = ast.parse(type_str, mode="eval").body
        call_node = ast.Call(func=func_node, args=[ast.Constant(value=node.value)], keywords=[])
        return ast.copy_location(call_node, node)


def safe_eval(
    expr: str,
    globals_vars: Mapping[str, Any] | None = None,
    locals_vars: Mapping[str, object] | None = None,
    allowed_types: Sequence[type] = SAFE_TYPES,
    rewrite_np: bool = False,
    int_type_str: str = "np.int32",
    float_type_str: str = "np.float32",
) -> Any:
    """
    Evaluate the Python expression `expr` using `eval`, but only if it is a safe expression in that its parsed AST
    contains nodes whose types are given in `allowed_types`. This ensures unsafe node types are excluded, if these
    are present in the AST a ValueError is raised. The default set of such types in `SAFE_TYPES` ensures only
    expressions with constants and names can be evaluated, so excludes attribute access, indexing, and calls. Code
    injection is infeasible through such expressions, so this is a safe and secure way of evaluating simple expressions.

    If `rewrite_np` is True, int and float constants in the given expression will be wrapped with Numpy types as given
    by `int_type_str` and `float_type_str`. These are expected to be constructor names prefixed with `np.` as Numpy
    will be present in the expression global variables under that name. The values can be changed to other types if
    needed, such as "int64". One advantage of doing this is to avoid denial-of-service attacks by attempting to evaluate
    an expression which is incredibly slow under native Python but fast (though potentially erroneous) under Numpy.

    Args:
        expr: expression to evaluate, this will be stripped before parsing to avoid indentation complaints
        globals_vars: global variable mapping, this will be treated as read-only for this function, unlike `eval`
        locals_vars: local variable mapping
        allowed_types: sequence of allowed AST types which can be found in `expr` when parsed
        rewrite_np: if True, wrap int or float literals in Numpy types
        int_type_str: int Numpy wrapping type string
        float_type_str: float Numpy wrapping type string

    Raises:
        ValueError: raised when any node in the AST parsed from `expr` has a type not in `allowed_types`

    Returns:
        The evaluated expression value, using `eval` with `globals_vars` and `locals_vars`
    """
    parsed = ast.parse(expr.strip(), mode="eval")

    # collect nodes in the AST which aren't permitted and unparse them for inclusion in the exception message
    disallowed = [ast.unparse(n) for n in ast.walk(parsed) if not isinstance(n, tuple(allowed_types))]

    if disallowed:
        raise ValueError(f"Unsafe expression `{expr}` not evaluated, contains disallowed components: {disallowed}")

    if rewrite_np:
        parsed = _RewriteConstNp(int_type_str, float_type_str).visit(parsed)
        ast.fix_missing_locations(parsed)
        locals_vars = {**(locals_vars or {}), "np": np}

    return eval(compile(parsed, "<safe_eval>", "eval"), dict(globals_vars) if globals_vars else None, locals_vars)
