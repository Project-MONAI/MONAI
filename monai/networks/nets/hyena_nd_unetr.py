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
HyenaNDUNETR: SwinUNETR with the HyenaND subquadratic operator in place of (or
mixed with) windowed self-attention.

This is a thin convenience subclass of :class:`monai.networks.nets.SwinUNETR` whose
defaults make Hyena placement explicit:

* ``use_hyena`` is forced to ``True``.
* ``hyena_stages`` is **required** -- callers must explicitly declare which of the
  four Swin stages run HyenaND vs windowed attention.

The classmethod :meth:`HyenaNDUNETR.get_variant` provides the three Hyena
variants from Table 4 of the NeurIPS 2026 paper "Native Multi-Dimensional Subquadratic
Operators via Input Dependent Long Convolutions" (paper id 26539):

==========  =================================  ==============================
Variant     ``hyena_stages``                   Notes
==========  =================================  ==============================
``HHHH``    ``(True,  True,  True,  True)``    Hyena at every Swin stage
``HAHA``    ``(True,  False, True,  False)``   striped/interleaved
``HHAA``    ``(True,  True,  False, False)``   paper-best (outer Hyena, inner attention)
==========  =================================  ==============================

``AAAA`` (pure attention) is intentionally not exposed here -- it is plain
:class:`SwinUNETR` and constructing a "HyenaNDUNETR" with no Hyena stages would be a
contradiction.

Requires the optional ``nvsubquadratic`` package; install with
``pip install monai[hyena]``.
"""

from __future__ import annotations

from collections.abc import Sequence

from monai.networks.nets.swin_unetr import SwinUNETR

__all__ = ["HyenaNDUNETR"]


# Per-stage Hyena patterns for the three paper variants.
PAPER_VARIANTS: dict[str, tuple[bool, ...]] = {
    "HHHH": (True, True, True, True),
    "HAHA": (True, False, True, False),
    "HHAA": (True, True, False, False),
}


class HyenaNDUNETR(SwinUNETR):
    """SwinUNETR with HyenaND replacing windowed self-attention at selected stages.

    See the module docstring for the paper-variant table and
    :meth:`get_variant` for a convenience constructor matching the NeurIPS
    2026 paper.  All other kwargs are forwarded to :class:`SwinUNETR`.

    Args:
        in_channels: dimension of input channels.
        out_channels: dimension of output channels.
        hyena_stages: required 4-tuple of bools, one per Swin stage. At least one
            element must be ``True`` (otherwise use :class:`SwinUNETR` directly).
        feature_size: dimension of network feature size. Must be a multiple of 12
            (inherited from :class:`SwinUNETR`).
        **kwargs: forwarded to :class:`SwinUNETR`.

    Raises:
        ValueError: if ``hyena_stages`` is missing, has the wrong length, or has no
            ``True`` element.
        ImportError: if the optional ``nvsubquadratic`` package is not installed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hyena_stages: Sequence[bool],
        feature_size: int = 48,
        **kwargs,
    ) -> None:
        if hyena_stages is None:
            raise ValueError(
                "HyenaNDUNETR requires `hyena_stages` (a 4-tuple of bools); "
                "use SwinUNETR directly for pure attention."
            )
        stages_tuple = tuple(bool(s) for s in hyena_stages)
        if len(stages_tuple) != 4:
            raise ValueError(
                f"hyena_stages must have length 4 (one bool per Swin stage); got length {len(stages_tuple)}."
            )
        if not any(stages_tuple):
            raise ValueError(
                "hyena_stages must enable HyenaND at at least one stage; "
                "use SwinUNETR directly for pure attention."
            )

        # ``use_hyena`` and ``hyena_stages`` are set explicitly here.  If the caller
        # also passed them through ``**kwargs`` we refuse the conflict rather than
        # silently override -- the subclass exists to make Hyena placement explicit.
        if "use_hyena" in kwargs:
            raise TypeError(
                "HyenaNDUNETR forces use_hyena=True; do not pass use_hyena via kwargs."
            )
        if "hyena_stages" in kwargs:
            raise TypeError(
                "HyenaNDUNETR receives hyena_stages as a positional/named arg; "
                "do not pass it again via kwargs."
            )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_hyena=True,
            hyena_stages=stages_tuple,
            **kwargs,
        )

    @classmethod
    def get_variant(cls, variant: str, **kwargs) -> "HyenaNDUNETR":
        """Build a :class:`HyenaNDUNETR` matching one of the NeurIPS 2026 paper variants.

        Args:
            variant: one of ``"HHHH"``, ``"HAHA"``, ``"HHAA"`` (case-insensitive).
            **kwargs: forwarded to :class:`HyenaNDUNETR.__init__`. Must include at
                least ``in_channels`` and ``out_channels``. Must NOT include
                ``hyena_stages`` (set by the variant).

        Returns:
            A :class:`HyenaNDUNETR` with ``hyena_stages`` set per the variant.

        Raises:
            ValueError: if ``variant`` is not one of the three known names, or if
                ``hyena_stages`` is also passed via kwargs.

        Example::

            >>> net = HyenaNDUNETR.get_variant(
            ...     "HHAA",
            ...     in_channels=1,
            ...     out_channels=29,
            ...     feature_size=48,
            ... )
        """
        key = variant.upper()
        if key not in PAPER_VARIANTS:
            raise ValueError(
                f"Unknown paper variant '{variant}'. "
                f"Known variants: {sorted(PAPER_VARIANTS)}. "
                "(AAAA is plain SwinUNETR; use that class directly.)"
            )
        if "hyena_stages" in kwargs:
            raise ValueError(
                "get_variant sets hyena_stages from the variant name; "
                "do not also pass hyena_stages via kwargs."
            )
        return cls(hyena_stages=PAPER_VARIANTS[key], **kwargs)
