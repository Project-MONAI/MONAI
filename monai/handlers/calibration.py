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

from collections.abc import Callable

from monai.handlers.ignite_metric import IgniteMetricHandler
from monai.metrics import CalibrationErrorMetric, CalibrationReduction
from monai.utils import MetricReduction

__all__ = ["CalibrationError"]


class CalibrationError(IgniteMetricHandler):
    """
    Ignite handler to compute Calibration Error during training or evaluation.

    **Why Calibration Matters:**

    A well-calibrated model produces probability estimates that match the true likelihood of correctness.
    For example, predictions with 80% confidence should be correct approximately 80% of the time.
    Modern neural networks often exhibit poor calibration (typically overconfident), which can be
    problematic in medical imaging where probability estimates may inform clinical decisions.

    This handler wraps :py:class:`~monai.metrics.CalibrationErrorMetric` for use with PyTorch Ignite
    engines, automatically computing and aggregating calibration errors across iterations.

    **Supported Calibration Metrics:**

    - **Expected Calibration Error (ECE)**: Weighted average of per-bin errors (most common).
    - **Average Calibration Error (ACE)**: Unweighted average across bins.
    - **Maximum Calibration Error (MCE)**: Worst-case calibration error.

    Args:
        num_bins: Number of equally-spaced bins for calibration computation. Defaults to 20.
        include_background: Whether to include the first channel (index 0) in computation.
            Set to ``False`` to exclude background in segmentation tasks. Defaults to ``True``.
        calibration_reduction: Calibration error reduction mode. Options: ``"expected"`` (ECE),
            ``"average"`` (ACE), ``"maximum"`` (MCE). Defaults to ``"expected"``.
        metric_reduction: Reduction across batch/channel after computing per-sample errors.
            Options: ``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``. Defaults to ``"mean"``.
        output_transform: Callable to extract ``(y_pred, y)`` from ``engine.state.output``.
            See `Ignite concepts <https://pytorch.org/ignite/concepts.html#state>`_ and
            the batch output transform tutorial in the MONAI tutorials repository.
        save_details: If ``True``, saves per-sample/per-channel metric values to
            ``engine.state.metric_details[name]``. Defaults to ``True``.

    References:
        - Guo, C., et al. "On Calibration of Modern Neural Networks." ICML 2017.
          https://proceedings.mlr.press/v70/guo17a.html
        - Barfoot, T., et al. "Average Calibration Losses for Reliable Uncertainty in
          Medical Image Segmentation." arXiv:2506.03942v3, 2025.
          https://arxiv.org/abs/2506.03942v3

    See Also:
        - :py:class:`~monai.metrics.CalibrationErrorMetric`: The underlying metric class.
        - :py:func:`~monai.metrics.calibration_binning`: Low-level binning for reliability diagrams.

    Example:
        >>> from monai.handlers import CalibrationError, from_engine
        >>> from ignite.engine import Engine
        >>>
        >>> def evaluation_step(engine, batch):
        ...     # Returns dict with "pred" (probabilities) and "label" (one-hot)
        ...     return {"pred": model(batch["image"]), "label": batch["label"]}
        >>>
        >>> evaluator = Engine(evaluation_step)
        >>>
        >>> # Attach calibration error handler
        >>> CalibrationError(
        ...     num_bins=15,
        ...     include_background=False,
        ...     calibration_reduction="expected",
        ...     output_transform=from_engine(["pred", "label"]),
        ... ).attach(evaluator, name="ECE")
        >>>
        >>> # After evaluation, access results
        >>> evaluator.run(val_loader)
        >>> ece = evaluator.state.metrics["ECE"]
        >>> print(f"Expected Calibration Error: {ece:.4f}")
    """

    def __init__(
        self,
        num_bins: int = 20,
        include_background: bool = True,
        calibration_reduction: CalibrationReduction | str = CalibrationReduction.EXPECTED,
        metric_reduction: MetricReduction | str = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        metric_fn = CalibrationErrorMetric(
            num_bins=num_bins,
            include_background=include_background,
            calibration_reduction=calibration_reduction,
            metric_reduction=metric_reduction,
        )

        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)
