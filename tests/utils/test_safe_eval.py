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
import unittest

import numpy as np
from parameterized import parameterized

from monai.utils import safe_eval

GOOD_EXPRS = [
    ("1+2", None, None, 3),
    ("     1    +     2   ", None, None, 3),
    ("1+2+x", {"x": 4}, None, 7),
    ("1+2+x", None, {"x": 4}, 7),
    ("1*2+x", {"x": 4}, None, 6),
    ("(1+2)*3", None, None, 9),
    ("foo+bar", {"foo": 1030}, {"bar": 204}, 1234),
]

BAD_EXPRS = [("foo()",), ("foo.bar",), ("foo[123]",), ("(1,2)",), ("[3,4]",), ("int.__class__.__init__.__globals__",)]


class TestSafeEval(unittest.TestCase):
    @parameterized.expand(GOOD_EXPRS)
    def test_good_exprs(self, expr, globals_vars, locals_vars, expected):
        """Test valid expressions with globals/locals evaluate to correct values."""
        result = safe_eval(expr, globals_vars, locals_vars)
        self.assertEqual(result, expected)

    @parameterized.expand(GOOD_EXPRS)
    def test_good_exprs_np(self, expr, globals_vars, locals_vars, expected):
        """Test valid expressions with globals/locals evaluate to correct values with Numpy wrapping."""
        result = safe_eval(expr, globals_vars, locals_vars, rewrite_np=True)
        self.assertEqual(result, expected)

    @parameterized.expand(BAD_EXPRS)
    def test_bad_exprs(self, expr):
        """Test bad expressions correctly raise ValueError."""
        with self.assertRaises(ValueError):
            safe_eval(expr)

        with self.assertRaises(ValueError):
            safe_eval(expr, rewrite_np=True)

    def test_allowed_types(self):
        """Test restricting the allowed list of types."""
        allowed = [ast.Expression, ast.Constant, ast.BinOp, ast.Add]
        result = safe_eval("1+2", allowed_types=allowed)
        self.assertEqual(result, 3)

        with self.assertRaises(ValueError):
            safe_eval("1*2", allowed_types=allowed)

    def test_rewrite_np_produces_numpy_types(self):
        """Test that rewrite_np wraps literals in numpy types."""
        result = safe_eval("2 + 3", rewrite_np=True)
        self.assertIsInstance(result, np.integer)

        result = safe_eval("2.5 + 1.5", rewrite_np=True)
        self.assertIsInstance(result, np.floating)

    def test_rewrite_np_large_exponent(self):
        """Test that rewrite_np prevents slow native-Python exponentiation."""
        # Under native Python, 9**9**9 produces a ~369-million-digit integer;
        # under np.int32 it overflows and completes almost instantly.
        result = safe_eval("9**9**9", rewrite_np=True)
        self.assertIsInstance(result, np.integer)

    def test_rewrite_np_preserves_bool(self):
        """Test that rewrite_np does not wrap bool constants."""
        result = safe_eval("True", rewrite_np=True)
        self.assertIs(result, True)

        result = safe_eval("False", rewrite_np=True)
        self.assertIs(result, False)

    def test_rewrite_np_inf_constant(self):
        """Test that rewrite_np handles overflowing infinity literals."""
        result = safe_eval("1e309", rewrite_np=True)
        self.assertIsInstance(result, np.floating)
        self.assertTrue(np.isinf(result))


if __name__ == "__main__":
    unittest.main()
