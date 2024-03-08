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

from typing import Callable

import torch
from torch import nn


def _zdot(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Complex dot product between tensors x1 and x2: sum(x1.*x2)
    """
    if torch.is_complex(x1):
        assert torch.is_complex(x2), "x1 and x2 must both be complex"
        return torch.sum(x1.conj() * x2)
    else:
        return torch.sum(x1 * x2)


def _zdot_single(x: torch.Tensor) -> torch.Tensor:
    """
    Complex dot product between tensor x and itself
    """
    res = _zdot(x, x)
    if torch.is_complex(res):
        return res.real
    else:
        return res


class ConjugateGradient(nn.Module):
    """
    Congugate Gradient (CG) solver for linear systems Ax = y.

    For linear_op that is positive definite and self-adjoint, CG is
    guaranteed to converge CG is often used to solve linear systems of the form
    Ax = y, where A is too large to store explicitly, but can be computed via a
    linear operator.

    As a result, here we won't set A explicitly as a matrix, but rather as a
    linear operator. For example, A could be a FFT/IFFT operation
    """

    def __init__(self, linear_op: Callable, num_iter: int):
        """
        Args:
            linear_op: Linear operator
            num_iter: Number of iterations to run CG
        """
        super().__init__()

        self.linear_op = linear_op
        self.num_iter = num_iter

    def update(
        self, x: torch.Tensor, p: torch.Tensor, r: torch.Tensor, rsold: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        perform one iteration of the CG method. It takes the current solution x,
        the current search direction p, the current residual r, and the old
        residual norm rsold as inputs. Then it computes the new solution, search
        direction, residual, and residual norm, and returns them.
        """

        dy = self.linear_op(p)
        p_dot_dy = _zdot(p, dy)
        alpha = rsold / p_dot_dy
        x = x + alpha * p
        r = r - alpha * dy
        rsnew = _zdot_single(r)
        beta = rsnew / rsold
        rsold = rsnew
        p = beta * p + r
        return x, p, r, rsold

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
        r = y - self.linear_op(x)
        rsold = _zdot_single(r)
        p = r

        # Update
        for _i in range(self.num_iter):
            x, p, r, rsold = self.update(x, p, r, rsold)
            if rsold < 1e-10:
                break
        return x
