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

from typing import Callable, Optional

import torch
from torch import nn


class ConjugateGradient(nn.Module):
    """
    Congugate Gradient (CG) solver for linear systems Ax = y.

    For A (linear_op) that is positive definite and self-adjoint, CG is
    guaranteed to converge CG is often used to solve linear systems of the form
    Ax = y, where A is too large to store explicitly, but can be computed via a
    linear operator.

    As a result, here we won't set A explicitly as a matrix, but rather as a
    linear operator. For example, A could be a FFT/IFFT operation
    """

    def __init__(self, linear_op: Callable, num_iter: int, dbprint: Optional[bool] = False):
        """
        Args:
            linear_op: Linear operator
            num_iter: Number of iterations to run CG
            dbprint [False]: Print residual at each iteration
        """
        super().__init__()

        self.A = linear_op
        self.num_iter = num_iter
        self.dbprint = dbprint

    def _zdot(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Complex dot product between tensors x1 and x2: sum(x1.*x2)
        """
        if torch.is_complex(x1):
            assert torch.is_complex(x2), "x1 and x2 must both be complex"
            return torch.sum(x1.conj() * x2)
        else:
            return torch.sum(x1 * x2)

    def _zdot_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complex dot product between tensor x and itself
        """
        res = self._zdot(x, x)
        if torch.is_complex(res):
            return res.real
        else:
            return res

    def _update(self, iter: int) -> Callable:
        """
        perform one iteration of the CG method. It takes the current solution x,
        the current search direction p, the current residual r, and the old
        residual norm rsold as inputs. Then it computes the new solution, search
        direction, residual, and residual norm, and returns them.
        """

        def update_fn(
            x: torch.Tensor, p: torch.Tensor, r: torch.Tensor, rsold: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            dy = self.A(p)
            p_dot_dy = self._zdot(p, dy)
            alpha = rsold / p_dot_dy
            x = x + alpha * p
            r = r - alpha * dy
            rsnew = self._zdot_single(r)
            beta = rsnew / rsold
            rsold = rsnew
            p = beta * p + r

            # print residual
            if self.dbprint:
                print(f"CG Iteration {iter}: {rsnew}")

            return x, p, r, rsold

        return update_fn

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        run conjugate gradient for num_iter iterations to solve Ax = y

        Args:
            x: tensor (real or complex); Initial guess for linear system Ax = y.
            The size of x should be applicable to the linear operator. For
            example, if the linear operator is FFT, then x is HCHW; if the
            linear operator is a matrix multiplication, then x is a vector

            y: tensor (real or complex); Measurement. Same size as x

        Returns:
            x: Solution to Ax = y
        """
        # Compute residual
        r = y - self.A(x)
        rsold = self._zdot_single(r)
        p = r

        # Update
        for i in range(self.num_iter):
            x, p, r, rsold = self._update(i)(x, p, r, rsold)
            if rsold < 1e-10:
                break
        return x
