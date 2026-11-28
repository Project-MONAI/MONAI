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

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetOutBlock, UnetResBlock, get_conv_layer
from monai.utils import ensure_tuple_rep, optional_import

Mamba, has_mamba_ssm = optional_import("mamba_ssm", name="Mamba")

__all__ = ["MambaBlock", "UMamba"]


class _MambaSSM(nn.Module):
    """
    Pure PyTorch implementation of the Mamba selective state space block.

    This is a fallback used when the optional ``mamba_ssm`` package is not installed. It follows the
    architecture of the reference implementation (input projection, causal depthwise convolution,
    input-dependent - "selective" - state space recurrence, gating, and output projection), but it
    evaluates the recurrence sequentially instead of using the fused CUDA kernel. This makes it run
    on any device, including CPU, at the cost of speed. For performance-critical training, install
    ``mamba_ssm`` (see https://github.com/state-spaces/mamba) and this block is replaced by the
    reference implementation automatically.

    Args:
        d_model: dimension of the input features (number of channels of the feature map).
        d_state: dimension of the state space latent state. Defaults to 16.
        d_conv: kernel size of the causal depthwise convolution. Defaults to 4.
        expand: expansion factor of the inner dimension. Defaults to 2.
        dt_rank: rank of the projection of the input-dependent step size, `"auto"` uses
            ``ceil(d_model / 16)``. Defaults to "auto".
        dt_min: minimum value of the step size after initialisation. Defaults to 0.001.
        dt_max: maximum value of the step size after initialisation. Defaults to 0.1.
        dt_init_floor: lower bound used when initialising the step-size bias. Defaults to 1e-4.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # initialise the dt bias so that softplus(dt_bias) lies in [dt_min, dt_max], as in mamba_ssm
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        a_mat = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_mat))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        _, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        # causal depthwise convolution over the sequence dimension
        x_in = self.conv1d(x_in.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_in = F.silu(x_in)
        y = self._selective_scan(x_in)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the input-dependent (selective) state space recurrence sequentially.

        Args:
            x: input tensor of shape (batch, seq_len, d_inner).

        Returns:
            Tensor of shape (batch, seq_len, d_inner).
        """
        batch, seq_len, _ = x.shape
        dt_b_c = self.x_proj(x)
        dt, b_mat, c_mat = torch.split(dt_b_c, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (batch, seq_len, d_inner)
        a_mat = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(seq_len):
            dt_t = dt[:, t]  # (batch, d_inner)
            h = h * torch.exp(dt_t.unsqueeze(-1) * a_mat) + (dt_t * x[:, t]).unsqueeze(-1) * b_mat[:, t].unsqueeze(1)
            outs.append(torch.sum(h * c_mat[:, t].unsqueeze(1), dim=-1))
        y = torch.stack(outs, dim=1)
        return y + x * self.D


class MambaBlock(nn.Module):
    """
    Mamba state space block for N-D feature maps, as used in U-Mamba.

    The feature map is flattened over its spatial dimensions into a sequence of tokens, normalised,
    processed by a Mamba block, and reshaped back to the original spatial dimensions. This captures
    long-range dependencies over the whole feature map with (near-)linear complexity in the number
    of voxels.

    If the optional ``mamba_ssm`` package is installed, its fused CUDA implementation is used for
    CUDA tensors. Otherwise (and for CPU tensors in general), a pure PyTorch implementation
    (:class:`_MambaSSM`) is used automatically, which runs on any device but evaluates the selective
    scan sequentially and is therefore slower.

    Args:
        dim: number of channels of the input feature map.
        d_state: dimension of the state space latent state. Defaults to 16.
        d_conv: kernel size of the local convolution inside the Mamba block. Defaults to 4.
        expand: expansion factor of the inner dimension of the Mamba block. Defaults to 2.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        if has_mamba_ssm:
            self.mamba: nn.Module = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            # the reference implementation is CUDA-only, so also keep the pure PyTorch fallback
            # and select between the two based on the input device in the forward pass
            self.mamba_fallback: nn.Module | None = _MambaSSM(
                d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            self.mamba = _MambaSSM(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_fallback = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input feature map of shape (batch, dim, *spatial).

        Returns:
            Tensor of the same shape and dtype as the input, with long-range dependencies modelled
            over the flattened spatial dimensions.
        """
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            if dtype in (torch.float16, torch.bfloat16):
                x = x.float()
            batch, channels = x.shape[:2]
            spatial_shape = x.shape[2:]
            x_flat = x.reshape(batch, channels, -1).transpose(1, 2)
            # the fused CUDA implementation only supports CUDA tensors, so run CPU tensors through
            # the pure PyTorch fallback instead
            mamba = self.mamba if x.is_cuda or self.mamba_fallback is None else self.mamba_fallback
            out = mamba(self.norm(x_flat))
            out = out.transpose(1, 2).reshape(batch, channels, *spatial_shape)
        return out.to(dtype)


class UMamba(nn.Module):
    """
    U-Mamba network for biomedical image segmentation, based on:
    `U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation
    <https://arxiv.org/abs/2401.04722>`_.

    U-Mamba combines residual convolutional blocks with Mamba state space blocks in a U-Net
    encoder-decoder with skip connections, so that the network models both local features and
    long-range dependencies. Two variants are supported through ``mamba_mode``:

    - ``"bot"`` (U-Mamba_Bot in the paper): a single Mamba block is applied to the bottleneck
      feature map. This is the default and the configuration recommended in most cases.
    - ``"enc"`` (U-Mamba_Enc in the paper): a Mamba block is applied after every encoder stage,
      so every skip connection carries state-space features into the decoder.

    Following the reference implementation, when the number of residual blocks per stage is given
    as a single integer, the stages in the deeper half of the encoder (and of the decoder) use a
    single residual block.

    The Mamba blocks use the fused CUDA implementation of the optional ``mamba_ssm`` package for
    CUDA tensors when it is installed, and a pure PyTorch implementation otherwise (see
    :class:`MambaBlock`), so the network can be built and run on CPU as well. Deep supervision
    follows the reference
    implementation: when ``deep_supervision`` is enabled, the forward pass returns a list of the
    segmentation outputs of every decoder stage (highest resolution first), each at its native
    resolution.

    Args:
        spatial_dims: number of spatial dimensions, should be 1, 2 or 3. Defaults to 3.
        in_channels: number of input channels. Defaults to 1.
        out_channels: number of output channels (for example, the number of segmentation classes).
            Defaults to 2.
        channels: number of feature channels at each encoder stage, given from the highest to the
            lowest resolution. Defaults to ``(16, 32, 64, 128, 256)``.
        strides: stride of the first residual block of each encoder stage, in the same order as
            ``channels``. The first entry is typically 1, and the decoder upsamples with the
            remaining strides. Defaults to ``(1, 2, 2, 2, 2)``.
        kernel_size: convolution kernel size of the residual blocks, either a single value used at
            every stage or one value per encoder stage. Defaults to 3.
        num_res_units: number of residual blocks per stage, either a single value or one value per
            encoder stage (the decoder uses the same number of stages minus one). Defaults to 2.
        act: activation type and arguments. Defaults to ``("LEAKYRELU", {"inplace": True,
            "negative_slope": 0.01})``.
        norm: feature normalisation type and arguments. Defaults to ``("INSTANCE", {"affine":
            True})``.
        dropout: dropout ratio. Defaults to no dropout.
        mamba_d_state: dimension of the state space latent state of the Mamba blocks. Defaults
            to 16.
        mamba_d_conv: kernel size of the local convolution inside the Mamba blocks. Defaults to 4.
        mamba_expand: expansion factor of the inner dimension of the Mamba blocks. Defaults to 2.
        mamba_mode: where the Mamba blocks are placed, ``"bot"`` for bottleneck-only (U-Mamba_Bot)
            or ``"enc"`` for every encoder stage (U-Mamba_Enc). Defaults to ``"bot"``.
        deep_supervision: whether to return the segmentation output of every decoder stage.
            Defaults to False.

    Example::

        import torch
        from monai.networks.nets import UMamba

        net = UMamba(spatial_dims=3, in_channels=1, out_channels=2)
        out = net(torch.randn(1, 1, 32, 32, 32))
        print(out.shape)  # (1, 2,
    ...[truncated 8243 chars] 32, 32, 32))
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        channels: Sequence[int] = (16, 32, 64, 128, 256),
        strides: Sequence[int] = (1, 2, 2, 2, 2),
        kernel_size: Sequence[int] | int = 3,
        num_res_units: int | Sequence[int] = 2,
        act: tuple | str = ("LEAKYRELU", {"inplace": True, "negative_slope": 0.01}),
        norm: tuple | str = ("INSTANCE", {"affine": True}),
        dropout: tuple | str | float | None = None,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_mode: str = "bot",
        deep_supervision: bool = False,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions, should be 1, 2 or 3. Defaults to 3.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels (for example, the number of segmentation
                classes). Defaults to 2.
            channels: number of feature channels at each encoder stage, given from the highest to
                the lowest resolution. Defaults to ``(16, 32, 64, 128, 256)``.
            strides: stride of the first residual block of each encoder stage, in the same order as
                ``channels``. Defaults to ``(1, 2, 2, 2, 2)``.
            kernel_size: convolution kernel size of the residual blocks, either a single value or
                one value per encoder stage. Defaults to 3.
            num_res_units: number of residual blocks per stage, either a single value or one value
                per encoder stage. A stage with 0 units applies a single convolution that performs
                the stage's channel and stride transition. Defaults to 2.
            act: activation type and arguments. Defaults to ``("LEAKYRELU", {"inplace": True,
                "negative_slope": 0.01})``.
            norm: feature normalisation type and arguments. Defaults to ``("INSTANCE", {"affine":
                True})``.
            dropout: dropout ratio. Defaults to no dropout.
            mamba_d_state: dimension of the state space latent state of the Mamba blocks. Defaults
                to 16.
            mamba_d_conv: kernel size of the local convolution inside the Mamba blocks. Defaults
                to 4.
            mamba_expand: expansion factor of the inner dimension of the Mamba blocks. Defaults
                to 2.
            mamba_mode: where the Mamba blocks are placed, ``"bot"`` for bottleneck-only
                (U-Mamba_Bot) or ``"enc"`` for every encoder stage (U-Mamba_Enc). Defaults to
                ``"bot"``.
            deep_supervision: whether to return the segmentation output of every decoder stage.
                Defaults to False.

        Raises:
            ValueError: if `spatial_dims` is not 1, 2 or 3.
            ValueError: if `mamba_mode` is not ``"bot"`` or ``"enc"``.
            ValueError: if `channels` has fewer than 2 stages.
            ValueError: if `strides` does not have the same number of entries as `channels`.
        """
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError(f"spatial_dims must be 1, 2 or 3, got {spatial_dims}.")
        if mamba_mode not in ("bot", "enc"):
            raise ValueError(f"mamba_mode must be 'bot' or 'enc', got '{mamba_mode}'.")
        if len(channels) < 2:
            raise ValueError(f"channels must have at least 2 stages, got {channels}.")
        if len(strides) != len(channels):
            raise ValueError(
                f"strides must have the same number of entries as channels ({len(channels)}), got {len(strides)}."
            )

        self.spatial_dims = spatial_dims
        self.mamba_mode = mamba_mode
        self.deep_supervision = deep_supervision
        n_stages = len(channels)

        kernel_sizes = self._per_stage(kernel_size, n_stages, "kernel_size")
        strides_rep = [ensure_tuple_rep(s, spatial_dims) for s in self._per_stage(strides, n_stages, "strides")]
        # per-dimension product of the stage strides: the input spatial size must be divisible by
        # this in every dimension for the encoder-decoder skip connections to line up (checked in
        # the forward pass)
        self.spatial_divisor = tuple(math.prod(stride[d] for stride in strides_rep) for d in range(spatial_dims))
        n_blocks_enc = self._per_stage(num_res_units, n_stages, "num_res_units")
        if isinstance(num_res_units, int):
            # the reference implementation uses a single residual block in the deeper half of
            # the encoder and decoder stages
            for s in range(math.ceil(n_stages / 2), n_stages):
                n_blocks_enc[s] = 1
            n_blocks_dec = [num_res_units] * (n_stages - 1)
            for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
                n_blocks_dec[s] = 1
        else:
            # decoder stage s reconstructs the resolution of encoder stage n_stages - 2 - s and
            # mirrors its number of residual blocks
            n_blocks_dec = [n_blocks_enc[n_stages - 2 - s] for s in range(n_stages - 1)]

        # encoder: residual blocks, the first block of each stage applies the stage stride
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        for s in range(n_stages):
            blocks: list[nn.Module] = []
            for b in range(n_blocks_enc[s]):
                blocks.append(
                    UnetResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=in_ch if b == 0 else channels[s],
                        out_channels=channels[s],
                        kernel_size=kernel_sizes[s],
                        stride=strides_rep[s] if b == 0 else 1,
                        norm_name=norm,
                        act_name=act,
                        dropout=dropout,
                    )
                )
            if not blocks:
                # a stage with no residual units still applies a single convolution that performs
                # the channel and stride transition of the stage
                blocks.append(
                    get_conv_layer(
                        spatial_dims=spatial_dims,
                        in_channels=in_ch,
                        out_channels=channels[s],
                        kernel_size=kernel_sizes[s],
                        stride=strides_rep[s],
                        act=None,
                        norm=None,
                        bias=True,
                        conv_only=True,
                    )
                )
            self.encoder.append(nn.Sequential(*blocks))
            in_ch = channels[s]

        # Mamba blocks
        if mamba_mode == "bot":
            self.bottleneck_mamba = MambaBlock(
                dim=channels[-1], d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand
            )
        else:
            self.mamba_layers = nn.ModuleList(
                [MambaBlock(dim=ch, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) for ch in channels]
            )

        # decoder: nearest-neighbour upsampling + 1x1 convolution (as in the reference), skip
        # connection by concatenation, then residual blocks; one segmentation head per stage
        self.upsample_layers = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for s in range(1, n_stages):
            in_below = channels[-s]
            in_skip = channels[-(s + 1)]
            self.upsample_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=strides_rep[-s], mode="nearest"),
                    get_conv_layer(
                        spatial_dims=spatial_dims,
                        in_channels=in_below,
                        out_channels=in_skip,
                        kernel_size=1,
                        stride=1,
                        act=None,
                        norm=None,
                        bias=True,
                        conv_only=True,
                    ),
                )
            )
            blocks = []
            for b in range(n_blocks_dec[s - 1]):
                blocks.append(
                    UnetResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=2 * in_skip if b == 0 else in_skip,
                        out_channels=in_skip,
                        kernel_size=kernel_sizes[-(s + 1)],
                        stride=1,
                        norm_name=norm,
                        act_name=act,
                        dropout=dropout,
                    )
                )
            if not blocks:
                # a stage with no residual units still applies a single convolution that maps the
                # concatenated skip input back to the stage channels
                blocks.append(
                    get_conv_layer(
                        spatial_dims=spatial_dims,
                        in_channels=2 * in_skip,
                        out_channels=in_skip,
                        kernel_size=kernel_sizes[-(s + 1)],
                        stride=1,
                        act=None,
                        norm=None,
                        bias=True,
                        conv_only=True,
                    )
                )
            self.decoder.append(nn.Sequential(*blocks))
            self.seg_layers.append(UnetOutBlock(spatial_dims, in_skip, out_channels, dropout=dropout))

    @staticmethod
    def _per_stage(value: Sequence[int] | int, n_stages: int, name: str) -> list:
        """
        Expand a per-stage argument to a list with one entry per stage.

        Args:
            value: a single value to repeat for every stage, or a sequence with one entry
                per stage.
            n_stages: number of stages the argument applies to.
            name: name of the argument, used in the error message.

        Returns:
            A list of length `n_stages` with the per-stage values.

        Raises:
            ValueError: if `value` is a sequence whose length differs from `n_stages`.
        """
        if isinstance(value, int):
            return [value] * n_stages
        value = list(value)
        if len(value) == n_stages:
            return value
        raise ValueError(f"{name} must be an int or a sequence of length {n_stages}, got {value}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """
        Args:
            x: input tensor of shape (batch, in_channels, *spatial). Every spatial dimension
                must be divisible by the product of the stage strides along that dimension.

        Returns:
            The segmentation output of shape (batch, out_channels, *spatial), or, when
            `deep_supervision` is enabled, a list with the segmentation output of every decoder
            stage (highest resolution first), each at its native resolution.

        Raises:
            ValueError: if a spatial dimension of `x` is not divisible by the product of the
                configured strides along that dimension.
        """
        spatial_size = tuple(x.shape[2:])
        for dim, (size, divisor) in enumerate(zip(spatial_size, self.spatial_divisor)):
            if size % divisor != 0:
                raise ValueError(
                    f"input spatial size {spatial_size} must be divisible by {self.spatial_divisor}, the "
                    f"product of the strides, in every dimension, but dimension {dim} has size {size}, "
                    f"which is not divisible by {divisor}."
                )
        skips: list[torch.Tensor] = []
        for s, stage in enumerate(self.encoder):
            x = stage(x)
            if self.mamba_mode == "enc":
                x = self.mamba_layers[s](x)
            skips.append(x)

        if self.mamba_mode == "bot":
            skips[-1] = self.bottleneck_mamba(skips[-1])

        seg_outputs: list[torch.Tensor] = []
        out = skips[-1]
        for s in range(len(self.decoder)):
            out = self.upsample_layers[s](out)
            out = torch.cat((out, skips[-(s + 2)]), dim=1)
            out = self.decoder[s](out)
            if self.deep_supervision or s == len(self.decoder) - 1:
                seg_outputs.append(self.seg_layers[s](out))

        if self.deep_supervision:
            return seg_outputs[::-1]
        return seg_outputs[0]
