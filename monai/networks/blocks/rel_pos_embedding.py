# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn

from monai.networks.blocks.attention_utils import add_decomposed_rel_pos
from monai.utils.misc import ensure_tuple_size


class DecomposedRelativePosEmbedding(nn.Module):
    def __init__(self, s_input_dims: Tuple[int, int] | Tuple[int, int, int], c_dim: int, num_heads: int) -> None:
        """
        Args:
            s_input_dims (Tuple): input spatial dimension. (H, W) or (H, W, D)
            c_dim (int): channel dimension
            num_heads(int): number of attention heads
        """
        super().__init__()

        # validate inputs
        if not isinstance(s_input_dims, Iterable) or len(s_input_dims) not in [2, 3]:
            raise ValueError("s_input_dims must be set as follows: (H, W) or (H, W, D)")

        self.s_input_dims = s_input_dims
        self.c_dim = c_dim
        self.num_heads = num_heads
        self.rel_pos_arr = nn.ParameterList(
            [nn.Parameter(torch.zeros(2 * dim_input_size - 1, c_dim)) for dim_input_size in s_input_dims]
        )

    def forward(self, x: torch.Tensor, att_mat: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """"""
        batch = x.shape[0]
        h, w, d = ensure_tuple_size(self.s_input_dims, 3, 1)

        att_mat = add_decomposed_rel_pos(
            att_mat.contiguous().view(batch * self.num_heads, h * w * d, h * w * d),
            q.contiguous().view(batch * self.num_heads, h * w * d, -1),
            self.rel_pos_arr,
            (h, w) if d == 1 else (h, w, d),
            (h, w) if d == 1 else (h, w, d),
        )

        att_mat = att_mat.reshape(batch, self.num_heads, h * w * d, h * w * d)
        return att_mat
