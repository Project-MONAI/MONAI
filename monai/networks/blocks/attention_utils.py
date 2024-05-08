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

from monai.utils import optional_import

rearrange, _ = optional_import("einops", name="rearrange")


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
    r"""
    Calculate decomposed Relative Positional Embeddings from mvitv2 implementation:
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Only 2D and 3D are supported.

    Encoding the relative position of tokens in the attention matrix: tokens spaced a distance
    `d` apart will have the same embedding value (unlike absolute positional embedding).

    .. math::
        Attn_{logits}(Q, K) = (QK^{T} + E_{rel})*scale

    where

    .. math::
        E_{ij}^{(rel)} = Q_{i}.R_{p(i), p(j)}

    with :math:`R_{p(i), p(j)} \in R^{dim}` and :math:`p(i), p(j)`,
    respectively spatial positions of element :math:`i` and :math:`j`

    When using "decomposed" relative positional embedding, positional embedding is defined ("decomposed") as follow:

    .. math::
        R_{p(i), p(j)} = R^{d1}_{d1(i), d1(j)} + ... + R^{dn}_{dn(i), dn(j)}

    with :math:`n = 1...dim`

    Decomposed relative positional embedding reduces the complexity from :math:`\mathcal{O}(d1*...*dn)` to
    :math:`\mathcal{O}(d1+...+dn)` compared with classical relative positional embedding.

    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, s_dim_1 * ... * s_dim_n, C).
        rel_pos_lst (ParameterList): relative position embeddings for each axis: rel_pos_lst[n] for nth axis.
        q_size (Tuple): spatial sequence size of query q with (q_dim_1, ..., q_dim_n).
        k_size (Tuple): spatial sequence size of key k with (k_dim_1, ...,  k_dim_n).

    Returns:
        attn (Tensor): attention logits with added relative positional embeddings.
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


def window_partition(x: torch.Tensor, window_size: int, input_size: Tuple = ()) -> Tuple[torch.Tensor, Tuple]:
    """
    Partition into non-overlapping windows with padding if needed. Support 2D and 3D.
    Args:
        x (tensor): input tokens with [B, s_dim_1 * ... * s_dim_n, C]. with n = 1...len(input_size)
        input_size (Tuple): input spatial dimension: (H, W) or (H, W, D)
        window_size (int): window size

    Returns:
        windows: windows after partition with [B * num_windows, window_size_1 * ... * window_size_n, C].
            with n = 1...len(input_size) and window_size_i == window_size.
        (S_DIM_1p, ...,S_DIM_np): padded spatial dimensions before partition with n = 1...len(input_size)
    """
    if x.shape[1] != int(torch.prod(torch.tensor(input_size))):
        raise ValueError(f"Input tensor spatial dimension {x.shape[1]} should be equal to {input_size} product")

    if len(input_size) == 2:
        x = rearrange(x, "b (h w) c -> b h w c", h=input_size[0], w=input_size[1])
        x, pad_hw = window_partition_2d(x, window_size)
        x = rearrange(x, "b h w c -> b (h w) c", h=window_size, w=window_size)
        return x, pad_hw
    elif len(input_size) == 3:
        x = rearrange(x, "b (h w d) c -> b h w d c", h=input_size[0], w=input_size[1], d=input_size[2])
        x, pad_hwd = window_partition_3d(x, window_size)
        x = rearrange(x, "b h w d c -> b (h w d) c", h=window_size, w=window_size, d=window_size)
        return x, pad_hwd
    else:
        raise ValueError(f"input_size cannot be length {len(input_size)}. It can be composed of 2 or 3 elements only. ")


def window_partition_2d(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed. Support only 2D.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    batch, h, w, c = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w

    x = x.view(batch, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, (hp, wp)


def window_partition_3d(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partition into non-overlapping windows with padding if needed. 3d implementation.
    Args:
        x (tensor): input tokens with [B, H, W, D, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, window_size, C].
        (Hp, Wp, Dp): padded height, width and depth before partition
    """
    batch, h, w, d, c = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    pad_d = (window_size - d % window_size) % window_size
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    hp, wp, dp = h + pad_h, w + pad_w, d + pad_d

    x = x.view(batch, hp // window_size, window_size, wp // window_size, window_size, dp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, c)
    return windows, (hp, wp, dp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad: Tuple, spatial_dims: Tuple) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size_1, ..., window_size_n, C].
            with n = 1...len(spatial_dims) and window_size == window_size_i
        window_size (int): window size.
        pad (Tuple): padded spatial dims (H, W) or (H, W, D)
        spatial_dims (Tuple): original spatial dimensions - (H, W) or (H, W, D) - before padding.

    Returns:
        x: unpartitioned sequences with [B, s_dim_1, ..., s_dim_n, C].
    """
    x: torch.Tensor
    if len(spatial_dims) == 2:
        x = rearrange(windows, "b (h w) c -> b h w c", h=window_size, w=window_size)
        x = window_unpartition_2d(x, window_size, pad, spatial_dims)
        x = rearrange(x, "b h w c -> b (h w) c", h=spatial_dims[0], w=spatial_dims[1])
        return x
    elif len(spatial_dims) == 3:
        x = rearrange(windows, "b (h w d) c -> b h w d c", h=window_size, w=window_size, d=window_size)
        x = window_unpartition_3d(x, window_size, pad, spatial_dims)
        x = rearrange(x, "b h w d c -> b (h w d) c", h=spatial_dims[0], w=spatial_dims[1], d=spatial_dims[2])
        return x
    else:
        raise ValueError()


def window_unpartition_2d(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (hp, wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    hp, wp = pad_hw
    h, w = hw
    batch = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(batch, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, hp, wp, -1)

    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


def window_unpartition_3d(
    windows: torch.Tensor, window_size: int, pad_hwd: Tuple[int, int, int], hwd: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding. 3d implementation.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, window_size, C].
        window_size (int): window size.
        pad_hwd (Tuple): padded height, width and depth (hp, wp, dp).
        hwd (Tuple): original height, width and depth (H, W, D) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, D, C].
    """
    hp, wp, dp = pad_hwd
    h, w, d = hwd
    batch = windows.shape[0] // (hp * wp * dp // window_size // window_size // window_size)
    x = windows.view(
        batch, hp // window_size, wp // window_size, dp // window_size, window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(batch, hp, wp, dp, -1)

    if hp > h or wp > w or dp > d:
        x = x[:, :h, :w, :d, :].contiguous()
    return x
