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

import unittest

import torch

from monai.networks.layers import ConjugateGradient


class TestConjugateGradient(unittest.TestCase):
    def test_real_valued_inverse(self):
        """Test ConjugateGradient with real-valued input: when the input is real value, the output should be the inverse of the matrix."""
        A_dim = 3
        A_mat = torch.tensor([[1, 2, 3], [2, 1, 2], [3, 2, 1]], dtype=torch.float)

        def A_op(x):
            return A_mat @ x

        cg_solver = ConjugateGradient(A_op, num_iter=100, dbprint=False)
        # define the measurement
        y = torch.tensor([1, 2, 3], dtype=torch.float)
        # solve for x
        x = cg_solver(torch.zeros(A_dim), y)
        x_ref = torch.linalg.solve(A_mat, y)
        # assert torch.allclose(x, x_ref, atol=1e-6), 'CG solver failed to converge to reference solution'
        self.assertTrue(torch.allclose(x, x_ref, atol=1e-6))
        print("real value test passed")

    def test_complex_valued_inverse(self):
        A_dim = 3
        A_mat = torch.tensor([[1, 2, 3], [2, 1, 2], [3, 2, 1]], dtype=torch.complex64)

        def A_op(x):
            return A_mat @ x

        cg_solver = ConjugateGradient(A_op, num_iter=100, dbprint=False)
        y = torch.tensor([1, 2, 3], dtype=torch.complex64)
        x = cg_solver(torch.zeros(A_dim, dtype=torch.complex64), y)
        x_ref = torch.linalg.solve(A_mat, y)
        self.assertTrue(torch.allclose(x, x_ref, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
