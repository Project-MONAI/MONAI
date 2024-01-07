# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    rel_pos_resized: torch.Tensor = torch.Tensor()
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode="linear"
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor, q: torch.Tensor, rel_pos_lst: nn.ParameterList, q_size: Tuple, k_size: Tuple
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Only 2D and 3D are supported.
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, s_dim_1 * ... * s_dim_n, C).
        rel_pos_lst (ParameterList): relative position embeddings for each axis: rel_pos_lst[n] for nth axis.
        q_size (Tuple): spatial sequence size of query q with (q_dim_1, ..., q_dim_n).
        k_size (Tuple): spatial sequence size of key k with (k_dim_1, ...,  k_dim_n).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    rh = get_rel_pos(q_size[0], k_size[0], rel_pos_lst[0])
    rw = get_rel_pos(q_size[1], k_size[1], rel_pos_lst[1])

    batch, _, dim = q.shape

    if len(rel_pos_lst) == 2:
        q_h, q_w = q_size[:2]
        k_h, k_w = k_size[:2]
        r_q = q.reshape(batch, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)

        attn = (attn.view(batch, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
            batch, q_h * q_w, k_h * k_w
        )
    elif len(rel_pos_lst) == 3:
        q_h, q_w, q_d = q_size[:3]
        k_h, k_w, k_d = k_size[:3]

        rd = get_rel_pos(q_d, k_d, rel_pos_lst[2])

        r_q = q.reshape(batch, q_h, q_w, q_d, dim)
        rel_h = torch.einsum("bhwdc,hkc->bhwdk", r_q, rh)
        rel_w = torch.einsum("bhwdc,wkc->bhwdk", r_q, rw)
        rel_d = torch.einsum("bhwdc,wkc->bhwdk", r_q, rd)

        attn = (
            attn.view(batch, q_h, q_w, q_d, k_h, k_w, k_d)
            + rel_h[:, :, :, :, None, None]
            + rel_w[:, :, :, None, :, None]
            + rel_d[:, :, :, None, None, :]
        ).view(batch, q_h * q_w * q_d, k_h * k_w * k_d)

    return attn
