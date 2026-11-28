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
HyenaND-based building blocks for MONAI networks.

These blocks provide a subquadratic O(N log N) alternative to windowed self-attention
in transformer-style segmentation networks. The operator is HyenaND from the
``nvsubquadratic`` package, gated through a thin :class:`HyenaMixer` and wrapped in
the conventional pre-norm / MLP residual pattern by :class:`HyenaTransformerBlock`.

The supporting :class:`DepthwiseFFTConv2d` / :class:`DepthwiseFFTConv3d` classes are
drop-in depthwise convolutions implemented via FFT. They preserve the
``nn.Conv{2,3}d`` weight layout and ``isinstance`` relationship but route the forward
pass through ``torch.fft.rfftn`` to avoid PyTorch's INT32 unfold limit, which caps
``F.conv3d`` at ROI ~128 for typical medical-imaging channel counts.

``nvsubquadratic`` is an optional dependency. The Hyena classes raise ``ImportError``
with an install hint at construction time if the library is unavailable; the
FFT-conv classes have no such dependency and always work.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from monai.networks.blocks.mlp import MLPBlock
from monai.networks.layers.drop_path import DropPath
from monai.utils import optional_import

__all__ = [
    "DepthwiseFFTConv2d",
    "DepthwiseFFTConv3d",
    "HyenaMixer",
    "HyenaTransformerBlock",
    "is_nvsubquadratic_available",
]

# Optional ``nvsubquadratic`` symbols.  Resolved at module-import time; the missing-dep
# error is raised lazily inside the consuming class ``__init__``.  Every symbol the Hyena
# classes use carries its own availability flag so a partial / broken install (e.g.
# ``lazy_config`` present but ``modules.hyena_nd`` missing) reports unavailable rather than
# failing later with an opaque ``AttributeError``.
_LazyConfig, _has_lazyconfig = optional_import("nvsubquadratic.lazy_config", name="LazyConfig")
_instantiate, _has_instantiate = optional_import("nvsubquadratic.lazy_config", name="instantiate")
_Hyena, _has_hyena = optional_import("nvsubquadratic.modules.hyena_nd", name="Hyena")
_CKConvND, _has_ckconv = optional_import("nvsubquadratic.modules.ckconv_nd", name="CKConvND")
_SIRENKernelND, _has_siren = optional_import("nvsubquadratic.modules.kernels_nd", name="SIRENKernelND")
_GaussianModulationND, _has_gaussian = optional_import("nvsubquadratic.modules.masks_nd", name="GaussianModulationND")
_has_nvsubq = all((_has_lazyconfig, _has_instantiate, _has_hyena, _has_ckconv, _has_siren, _has_gaussian))

_NVSUBQ_INSTALL_HINT = (
    "HyenaND operators require the optional 'nvsubquadratic' package (Python >= 3.10). "
    "Install it with:  pip install 'monai[hyena]'  "
    "(equivalently: pip install 'nvsubquadratic>=0.1.1'). "
    "See https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies"
)


def is_nvsubquadratic_available() -> bool:
    """Return ``True`` if the optional ``nvsubquadratic`` package is importable."""
    return bool(_has_nvsubq)


# ---------------------------------------------------------------------------
# Depthwise FFT convolutions — no nvsubquadratic dependency
# ---------------------------------------------------------------------------


class _DepthwiseFFTForward:
    """Mixin providing FFT-based forward for depthwise ``nn.Conv{2,3}d`` subclasses.

    No ``nn.Module`` parent: module machinery comes from ``nn.Conv2d`` / ``nn.Conv3d``
    in the concrete subclasses. Placed first in the MRO so ``forward`` resolves here
    (FFT) rather than to ``nn.Conv{2,3}d.forward`` (im2col / unfold).

    Avoids PyTorch's im2col INT32 overflow, which caps ``F.conv3d`` at ROI ~128 for
    typical medical-imaging channel counts. There is no spatial-size restriction.

    ``fft_chunk_size > 0`` enables channel-chunked FFT to cap peak memory:

        peak ≈ (B × chunk × spatial × 4 + B × chunk × rfft_spatial × 8) bytes

    instead of the full ``(B × C × ...)`` allocation.
    """

    _spatial_dims: int  # set by subclasses
    fft_chunk_size: int = 0  # 0 = no chunking; set in subclass __init__

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        spatial = input.shape[2:]
        kernel_shape = self.weight.shape[2:]  # type: ignore[attr-defined]
        fft_dims = tuple(range(-self._spatial_dims, 0))
        fft_size = [s + k - 1 for s, k in zip(spatial, kernel_shape)]
        in_dtype = input.dtype

        slices = (slice(None), slice(None)) + tuple(slice(k // 2, k // 2 + s) for s, k in zip(spatial, kernel_shape))

        chunk = getattr(self, "fft_chunk_size", 0)
        if chunk > 0 and input.shape[1] > chunk:
            parts = []
            for c0 in range(0, input.shape[1], chunk):
                c1 = min(c0 + chunk, input.shape[1])
                xc = input[:, c0:c1].float()
                kc = self.weight[c0:c1].squeeze(1).float()  # type: ignore[attr-defined]
                kc = kc.flip(list(range(1, self._spatial_dims + 1)))
                xc_fft = torch.fft.rfftn(xc, s=fft_size, dim=fft_dims)
                kc_fft = torch.fft.rfftn(kc, s=fft_size, dim=fft_dims)
                out_fft = xc_fft * kc_fft.unsqueeze(0)
                del xc_fft, kc_fft
                out_c = torch.fft.irfftn(out_fft, s=fft_size, dim=fft_dims)
                del out_fft
                parts.append(out_c[slices].to(in_dtype))
                del out_c
            return torch.cat(parts, dim=1)

        x_f32 = input.float()
        k_f32 = self.weight.squeeze(1).float()  # type: ignore[attr-defined]
        # PyTorch ``F.conv*`` computes cross-correlation; FFT computes convolution.
        # Flip the kernel so the FFT output matches ``Conv{2,3}d`` exactly.
        k_f32 = k_f32.flip(list(range(1, self._spatial_dims + 1)))

        x_fft = torch.fft.rfftn(x_f32, s=fft_size, dim=fft_dims)
        k_fft = torch.fft.rfftn(k_f32, s=fft_size, dim=fft_dims)

        out_fft = x_fft * k_fft.unsqueeze(0)
        out = torch.fft.irfftn(out_fft, s=fft_size, dim=fft_dims)
        return cast(torch.Tensor, out[slices].to(in_dtype))


def _validate_depthwise_fft_args(
    in_channels: int, out_channels: int, kernel_size: int, groups: int, padding: int, bias: bool
) -> None:
    """Validate the constructor arguments shared by ``DepthwiseFFTConv{2,3}d``.

    The FFT forward only implements depthwise, bias-free, ``"same"``-style convolution: it
    crops the full convolution back to the input spatial size assuming ``padding ==
    kernel_size // 2`` with an odd kernel. Reject anything else up front rather than return a
    silently wrong shape.
    """
    if not (in_channels == out_channels == groups):
        raise ValueError(
            "DepthwiseFFTConv only supports depthwise (groups == in_channels == out_channels); "
            f"got in_channels={in_channels}, out_channels={out_channels}, groups={groups}"
        )
    if bias:
        raise ValueError("bias is not supported in DepthwiseFFTConv")
    if kernel_size % 2 == 0 or padding != kernel_size // 2:
        raise ValueError(
            "DepthwiseFFTConv only supports 'same'-style padding: kernel_size must be odd and "
            f"padding must equal kernel_size // 2; got kernel_size={kernel_size}, padding={padding}. "
            "The FFT forward crops to the input spatial size and does not implement general padding."
        )


class DepthwiseFFTConv2d(_DepthwiseFFTForward, nn.Conv2d):
    """2-D depthwise FFT convolution. ``isinstance(x, nn.Conv2d)`` remains ``True``.

    Drop-in replacement for an ``nn.Conv2d`` with ``groups == in_channels == out_channels``
    and ``bias=False``. Useful as the short-conv inside :class:`HyenaMixer` at large 2-D
    inputs, where ``F.conv2d`` would not yet hit the INT32 limit but the unified
    Conv2d/Conv3d API is convenient.

    Only ``"same"``-style padding is supported: ``padding`` must equal ``kernel_size // 2``
    (and ``kernel_size`` must be odd). The FFT forward crops its output back to the input
    spatial size and does not implement general (e.g. ``"valid"``) padding.
    """

    _spatial_dims = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        padding: int,
        bias: bool = False,
        fft_chunk_size: int = 0,
    ) -> None:
        _validate_depthwise_fft_args(in_channels, out_channels, kernel_size, groups, padding, bias)
        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=padding, groups=groups, bias=False
        )
        self.fft_chunk_size = fft_chunk_size


class DepthwiseFFTConv3d(_DepthwiseFFTForward, nn.Conv3d):
    """3-D depthwise FFT convolution. ``isinstance(x, nn.Conv3d)`` remains ``True``.

    Drop-in replacement for an ``nn.Conv3d`` with ``groups == in_channels == out_channels``
    and ``bias=False``. Avoids the INT32 unfold limit that prevents ``F.conv3d`` from
    running at ROI > ~128 for typical medical-imaging channel counts.

    Only ``"same"``-style padding is supported: ``padding`` must equal ``kernel_size // 2``
    (and ``kernel_size`` must be odd). The FFT forward crops its output back to the input
    spatial size and does not implement general (e.g. ``"valid"``) padding.
    """

    _spatial_dims = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        padding: int,
        bias: bool = False,
        fft_chunk_size: int = 0,
    ) -> None:
        _validate_depthwise_fft_args(in_channels, out_channels, kernel_size, groups, padding, bias)
        nn.Conv3d.__init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=padding, groups=groups, bias=False
        )
        self.fft_chunk_size = fft_chunk_size


# ---------------------------------------------------------------------------
# HyenaMixer — QKV-projected gated long-conv mixer
# ---------------------------------------------------------------------------


class HyenaMixer(nn.Module):
    """QKV-projected gated long-convolution mixer using HyenaND.

    Replaces self-attention with the HyenaND operator from ``nvsubquadratic``, providing
    a global receptive field at O(N log N) cost via FFT. ``HyenaMixer`` matches the
    channels-last layout ``[B, *spatial, C]`` expected by Swin-style transformer
    blocks; the underlying HyenaND operator handles the 2-D / 3-D distinction.

    The HyenaND FFT path requires float32 precision, so the inner mixer call is
    wrapped in ``torch.amp.autocast("cuda", enabled=False)``. Inputs are cast to
    float32 before the mixer call and back to the original dtype after, so the block
    is transparent under ``torch.autocast``.

    Args:
        dim: hidden dimension.
        spatial_dims: 2 or 3.
        use_rope: kept for forward-compatibility with older configurations.
            ``nvsubquadratic`` removed RoPE from HyenaND on 2026-04-27 and this kwarg
            is now a silent no-op.
        apply_qk_norm: whether to apply per-channel LayerNorm to Q (and to K when the
            first gate is the identity, as is the default here).
        short_conv_kernel_size: depthwise short-convolution kernel size on the
            concatenated ``[Q; K; V]`` tensor.
        kernel_mlp_hidden_dim: hidden dim of the SIREN implicit kernel MLP.
        kernel_num_layers: depth of the SIREN implicit kernel.
        kernel_omega_0: SIREN frequency. Default 10.0 (stable). Higher values allow
            higher-frequency kernels but reduce training stability.
        kernel_l_cache: SIREN coordinate-grid cache size per spatial dim. Memory
            scales as ``(2L-1) ** D × D × 4`` bytes; set to ``>= max(spatial_dims)``
            to pre-allocate.
        mask_max_attenuation: Gaussian-modulation attenuation at the grid boundary
            for the widest channel (0–1). Default 0.95.
        fft_padding: ``"circular"`` or ``"zero"``.
        grid_type: ``"single"`` (kernel size = input size; required for circular
            padding) or ``"double"`` (kernel size = 2× input size; only valid with
            ``"zero"`` padding).
        use_chunked_fftconv: chunk the FFT convolution by channel to reduce peak
            memory ~26% with ~11% compute overhead. Requires ``fft_padding="zero"``.
        use_fft_short_conv: replace the depthwise short conv with
            :class:`DepthwiseFFTConv{2,3}d`, eliminating the INT32 unfold limit and
            enabling unlimited ROI sizes. Adds ~11% compute overhead.
        short_conv_fft_chunk_size: channel chunk size for the FFT short conv
            (0 = no chunking).

    Raises:
        ImportError: if ``nvsubquadratic`` is not installed.
        ValueError: on invalid ``spatial_dims`` / ``fft_padding`` / ``grid_type``
            combinations.
    """

    def __init__(
        self,
        dim: int,
        spatial_dims: int = 3,
        use_rope: bool = True,
        apply_qk_norm: bool = True,
        short_conv_kernel_size: int = 3,
        kernel_mlp_hidden_dim: int = 32,
        kernel_num_layers: int = 3,
        kernel_omega_0: float = 10.0,
        kernel_l_cache: int = 32,
        mask_max_attenuation: float = 0.95,
        fft_padding: str = "circular",
        grid_type: str = "single",
        use_chunked_fftconv: bool = False,
        use_fft_short_conv: bool = False,
        short_conv_fft_chunk_size: int = 0,
    ) -> None:
        super().__init__()

        if not _has_nvsubq:
            raise ImportError(_NVSUBQ_INSTALL_HINT)

        if fft_padding not in ("circular", "zero"):
            raise ValueError(f"fft_padding must be 'circular' or 'zero', got '{fft_padding}'")
        if grid_type not in ("single", "double"):
            raise ValueError(f"grid_type must be 'single' or 'double', got '{grid_type}'")
        if fft_padding == "circular" and grid_type != "single":
            raise ValueError(
                "fft_padding='circular' requires grid_type='single' "
                f"(kernel size must match input size for periodic convolution); got grid_type='{grid_type}'"
            )
        if use_chunked_fftconv and fft_padding != "zero":
            raise ValueError(
                "use_chunked_fftconv=True requires fft_padding='zero'; " f"got fft_padding='{fft_padding}'"
            )

        self.dim = dim
        self.spatial_dims = spatial_dims

        conv_class: type[nn.Module]
        if use_fft_short_conv:
            if spatial_dims == 2:
                conv_class = DepthwiseFFTConv2d
            elif spatial_dims == 3:
                conv_class = DepthwiseFFTConv3d
            else:
                raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
        else:
            if spatial_dims == 2:
                conv_class = nn.Conv2d
            elif spatial_dims == 3:
                conv_class = nn.Conv3d
            else:
                raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

        global_conv_cfg = _LazyConfig(_CKConvND)(
            data_dim=spatial_dims,
            hidden_dim=dim,
            kernel_cfg=_LazyConfig(_SIRENKernelND)(
                data_dim=spatial_dims,
                out_dim=dim,
                mlp_hidden_dim=kernel_mlp_hidden_dim,
                num_layers=kernel_num_layers,
                embedding_dim=kernel_mlp_hidden_dim,
                omega_0=kernel_omega_0,
                L_cache=kernel_l_cache,
                use_bias=True,
                hidden_omega_0=1.0,
            ),
            mask_cfg=_LazyConfig(_GaussianModulationND)(
                data_dim=spatial_dims,
                num_channels=dim,
                min_attenuation_at_step=0.1,
                max_attenuation_at_limit=mask_max_attenuation,
                init_extent=1.0,
                parametrization="direct",
            ),
            grid_type=grid_type,
            fft_padding=fft_padding,
            use_chunked_fftconv=use_chunked_fftconv,
        )

        short_conv_kwargs: dict = dict(
            in_channels=3 * dim,
            out_channels=3 * dim,
            kernel_size=short_conv_kernel_size,
            groups=3 * dim,
            padding=short_conv_kernel_size // 2,
            bias=False,
        )
        if use_fft_short_conv and short_conv_fft_chunk_size > 0:
            short_conv_kwargs["fft_chunk_size"] = short_conv_fft_chunk_size
        short_conv_cfg = _LazyConfig(conv_class)(**short_conv_kwargs)

        # ``use_rope`` retained on the API only; ``nvsubquadratic`` removed RoPE
        # from ``Hyena.__init__`` on 2026-04-27. Saving the flag here keeps caller
        # introspection intact ("did the user ask for RoPE?") while not affecting
        # the constructed operator.
        self._use_rope_requested = use_rope

        self.mixer = _instantiate(
            _LazyConfig(_Hyena)(
                global_conv_cfg=global_conv_cfg,
                short_conv_cfg=short_conv_cfg,
                gate_nonlinear_cfg=_LazyConfig(nn.Identity)(),
                pixelhyena_norm_cfg=_LazyConfig(nn.GroupNorm)(num_groups=1, num_channels=dim),
                qk_norm_cfg=_LazyConfig(nn.LayerNorm)(normalized_shape=dim) if apply_qk_norm else None,
            )
        )

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor of shape ``[batch, *spatial, dim]``.

        Returns:
            Tensor of the same shape and dtype as the input.
        """
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # HyenaND requires float32 internally; disable autocast for the mixer only,
        # then restore the original (autocast) dtype afterwards.
        with torch.amp.autocast("cuda", enabled=False):
            q = q.float()
            k = k.float()
            v = v.float()
            x = self.mixer(q, k, v)
        x = x.to(qkv.dtype)
        return cast(torch.Tensor, self.out_proj(x))


# ---------------------------------------------------------------------------
# HyenaTransformerBlock — pre-norm Hyena + MLP residual
# ---------------------------------------------------------------------------


class HyenaTransformerBlock(nn.Module):
    """Pre-norm transformer block with HyenaND in place of self-attention.

    Sandwiches a :class:`HyenaMixer` and an MLP between :class:`~torch.nn.LayerNorm`
    layers in the standard transformer residual pattern, with optional gradient
    checkpointing on each half.

    Args:
        dim: number of feature channels.
        spatial_dims: 2 or 3.
        mlp_ratio: hidden / input ratio for the MLP.
        drop: dropout rate inside the MLP.
        drop_path: stochastic-depth rate for the MLP residual.
        act_layer: activation name passed to :class:`monai.networks.blocks.MLPBlock`.
        norm_layer: normalization class (default :class:`~torch.nn.LayerNorm`).
        use_checkpoint: enable gradient checkpointing on the mixer and MLP halves.
        use_rope: forward-compatibility flag for older configs; no-op after the
            ``nvsubquadratic`` 2026-04-27 RoPE removal.
        apply_qk_norm: per-channel LayerNorm on Q (and K when the first gate is
            identity, as is the default).
        hyena_kernel_size: short-convolution kernel size on the QKV tensor.
        hyena_kernel_mlp_dim: SIREN kernel MLP hidden dimension.
        hyena_kernel_layers: SIREN kernel depth.
        hyena_mask_max_attenuation: Gaussian-modulation boundary attenuation (0–1).
        hyena_fft_padding: ``"circular"`` or ``"zero"``.
        hyena_grid_type: ``"single"`` or ``"double"``.
        hyena_use_chunked_fft: enable chunked FFT (requires zero padding).
        hyena_use_fft_short_conv: use FFT for the short conv (no INT32 limit).
        hyena_omega_0: SIREN ``omega_0``. Default 10.0.
        hyena_l_cache: SIREN coordinate-grid cache size per dim.
        hyena_short_conv_fft_chunks: channel chunk size for the FFT short conv.

    Raises:
        ImportError: if ``nvsubquadratic`` is not installed.
    """

    def __init__(
        self,
        dim: int,
        spatial_dims: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,  # type: ignore[assignment]
        use_checkpoint: bool = False,
        use_rope: bool = True,
        apply_qk_norm: bool = True,
        hyena_kernel_size: int = 3,
        hyena_kernel_mlp_dim: int = 32,
        hyena_kernel_layers: int = 3,
        hyena_mask_max_attenuation: float = 0.95,
        hyena_fft_padding: str = "circular",
        hyena_grid_type: str = "single",
        hyena_use_chunked_fft: bool = False,
        hyena_use_fft_short_conv: bool = False,
        hyena_omega_0: float = 10.0,
        hyena_l_cache: int = 32,
        hyena_short_conv_fft_chunks: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.spatial_dims = spatial_dims
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.mixer = HyenaMixer(
            dim=dim,
            spatial_dims=spatial_dims,
            use_rope=use_rope,
            apply_qk_norm=apply_qk_norm,
            short_conv_kernel_size=hyena_kernel_size,
            kernel_mlp_hidden_dim=hyena_kernel_mlp_dim,
            kernel_num_layers=hyena_kernel_layers,
            kernel_omega_0=hyena_omega_0,
            kernel_l_cache=hyena_l_cache,
            mask_max_attenuation=hyena_mask_max_attenuation,
            fft_padding=hyena_fft_padding,
            grid_type=hyena_grid_type,
            use_chunked_fftconv=hyena_use_chunked_fft,
            use_fft_short_conv=hyena_use_fft_short_conv,
            short_conv_fft_chunk_size=hyena_short_conv_fft_chunks,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPBlock(
            hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin"
        )

    def forward_part1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        return cast(torch.Tensor, self.mixer(x))

    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.drop_path(self.mlp(self.norm2(x))))

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape ``[batch, *spatial, dim]``.
            mask_matrix: unused; accepted for signature parity with Swin's
                ``WindowAttention``-based block so the two can be swapped at the
                ``BasicLayer`` level without per-call branching.

        Returns:
            Tensor of the same shape as the input.
        """
        del mask_matrix
        shortcut = x
        if self.use_checkpoint:
            x = torch.utils.checkpoint.checkpoint(self.forward_part1, x, use_reentrant=False)
        else:
            x = self.forward_part1(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + torch.utils.checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x
