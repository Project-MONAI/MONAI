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

import warnings
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import torch

from monai.config import NdarrayOrTensor, NdarrayTensor
from monai.transforms.croppad.dictionary import CropForegroundD
from monai.utils import (
    MetricReduction,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple_rep,
    look_up_option,
    optional_import,
)

binary_erosion, _ = optional_import("scipy.ndimage.morphology", name="binary_erosion")
distance_transform_edt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_edt")
distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")

__all__ = [
    "ignore_background",
    "do_metric_reduction",
    "get_mask_edges",
    "get_surface_distance",
    "is_binary_tensor",
    "remap_instance_id",
    "prepare_spacing",
    "get_code_to_measure_table",
]


def ignore_background(y_pred: NdarrayTensor, y: NdarrayTensor) -> tuple[NdarrayTensor, NdarrayTensor]:
    """
    This function is used to remove background (the first channel) for `y_pred` and `y`.

    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        y: ground truth, the first dim is batch.

    """

    y = y[:, 1:] if y.shape[1] > 1 else y  # type: ignore[assignment]
    y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred  # type: ignore[assignment]
    return y_pred, y


def do_metric_reduction(
    f: torch.Tensor, reduction: MetricReduction | str = MetricReduction.MEAN
) -> tuple[torch.Tensor | Any, torch.Tensor]:
    """
    This function is to do the metric reduction for calculated `not-nan` metrics of each sample's each class.
    The function also returns `not_nans`, which counts the number of not nans for the metric.

    Args:
        f: a tensor that contains the calculated metric scores per batch and
            per class. The first two dims should be batch and class.
        reduction: define the mode to reduce metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``.
            if "none", return the input f tensor and not_nans.

    Raises:
        ValueError: When ``reduction`` is not one of
            ["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].
    """

    # some elements might be Nan (if ground truth y was missing (zeros))
    # we need to account for it
    nans = torch.isnan(f)
    not_nans = (~nans).float()

    t_zero = torch.zeros(1, device=f.device, dtype=f.dtype)
    reduction = look_up_option(reduction, MetricReduction)
    if reduction == MetricReduction.NONE:
        return f, not_nans

    f[nans] = 0
    if reduction == MetricReduction.MEAN:
        # 2 steps, first, mean by channel (accounting for nans), then by batch
        not_nans = not_nans.sum(dim=1)
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average

        not_nans = (not_nans > 0).float().sum(dim=0)
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average

    elif reduction == MetricReduction.SUM:
        not_nans = not_nans.sum(dim=[0, 1])
        f = torch.sum(f, dim=[0, 1])  # sum over the batch and channel dims
    elif reduction == MetricReduction.MEAN_BATCH:
        not_nans = not_nans.sum(dim=0)
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average
    elif reduction == MetricReduction.SUM_BATCH:
        not_nans = not_nans.sum(dim=0)
        f = f.sum(dim=0)  # the batch sum
    elif reduction == MetricReduction.MEAN_CHANNEL:
        not_nans = not_nans.sum(dim=1)
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average
    elif reduction == MetricReduction.SUM_CHANNEL:
        not_nans = not_nans.sum(dim=1)
        f = f.sum(dim=1)  # the channel sum
    elif reduction != MetricReduction.NONE:
        raise ValueError(
            f"Unsupported reduction: {reduction}, available options are "
            '["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].'
        )
    return f, not_nans


def get_mask_edges(
    seg_pred: NdarrayOrTensor,
    seg_gt: NdarrayOrTensor,
    label_idx: int = 1,
    crop: bool = True,
    spacing: Sequence | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute edges from binary segmentation masks. This
    function is helpful to further calculate metrics such as Average Surface
    Distance and Hausdorff Distance.
    The input images can be binary or labelfield images. If labelfield images
    are supplied, they are converted to binary images using `label_idx`.

    In order to improve the computing efficiency, before getting the edges,
    the images can be cropped and only keep the foreground if not specifies
    ``crop = False``.

    We require that images are the same size, and assume that they occupy the
    same space (spacing, orientation, etc.).

    Args:
        seg_pred: the predicted binary or labelfield image.
        seg_gt: the actual binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        crop: crop input images and only keep the foregrounds. In order to
            maintain two inputs' shapes, here the bounding box is achieved
            by ``(seg_pred | seg_gt)`` which represents the union set of two
            images. Defaults to ``True``.
        spacing: the input spacing. If not None, the subvoxel edges and areas will be computed.
            otherwise `scipy`'s binary erosion is used to calculate the edges.
    """
    if seg_pred.shape != seg_gt.shape:
        raise ValueError(f"seg_pred and seg_gt should have same shapes, got {seg_pred.shape} and {seg_gt.shape}.")

    # If not binary images, convert them
    if seg_pred.dtype not in (bool, torch.bool):
        seg_pred = seg_pred == label_idx
    if seg_gt.dtype not in (bool, torch.bool):
        seg_gt = seg_gt == label_idx
    if crop:
        or_vol = seg_pred | seg_gt
        if not or_vol.any():
            pred, gt = np.zeros(seg_pred.shape, dtype=bool), np.zeros(seg_gt.shape, dtype=bool)
            return (pred, gt) if spacing is None else (pred, gt, pred, gt)  # type: ignore
        channel_first = [seg_pred[None], seg_gt[None], or_vol[None]]
        if spacing is None:  # cpu only erosion
            seg_pred, seg_gt, or_vol = convert_to_tensor(channel_first, device="cpu", dtype=bool)
        else:  # pytorch subvoxel, maybe on gpu, but croppad boolean values on GPU is not supported
            seg_pred, seg_gt, or_vol = convert_to_tensor(channel_first, dtype=torch.float16)
        cropper = CropForegroundD(
            ["pred", "gt"], source_key="src", margin=1, allow_smaller=False, start_coord_key=None, end_coord_key=None
        )
        cropped = cropper({"pred": seg_pred, "gt": seg_gt, "src": or_vol})  # type: ignore
        seg_pred, seg_gt = cropped["pred"][0], cropped["gt"][0]

    if spacing is None:  # Do binary erosion and use XOR to get edges
        seg_pred, seg_gt = convert_to_numpy([seg_pred, seg_gt], dtype=bool)
        edges_pred = binary_erosion(seg_pred) ^ seg_pred
        edges_gt = binary_erosion(seg_gt) ^ seg_gt
        return edges_pred, edges_gt
    code_to_area_table, k = get_code_to_measure_table(spacing, device=seg_pred.device)  # type: ignore
    spatial_dims = len(spacing)
    conv = torch.nn.functional.conv3d if spatial_dims == 3 else torch.nn.functional.conv2d
    vol = torch.stack([seg_pred[None], seg_gt[None]], dim=0).float()  # type: ignore
    code_pred, code_gt = conv(vol, k.to(vol))
    # edges
    all_ones = len(code_to_area_table) - 1
    edges_pred = (code_pred != 0) & (code_pred != all_ones)
    edges_gt = (code_gt != 0) & (code_gt != all_ones)
    # areas of edges
    areas_pred = torch.index_select(code_to_area_table, 0, code_pred.view(-1).int()).reshape(code_pred.shape)
    areas_gt = torch.index_select(code_to_area_table, 0, code_gt.view(-1).int()).reshape(code_gt.shape)
    ret = (edges_pred[0], edges_gt[0], areas_pred[0], areas_gt[0])
    return convert_to_numpy(ret, wrap_sequence=False)  # type: ignore


def get_surface_distance(
    seg_pred: np.ndarray,
    seg_gt: np.ndarray,
    distance_metric: str = "euclidean",
    spacing: int | float | np.ndarray | Sequence[int | float] | None = None,
) -> np.ndarray:
    """
    This function is used to compute the surface distances from `seg_pred` to `seg_gt`.

    Args:
        seg_pred: the edge of the predictions.
        seg_gt: the edge of the ground truth.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.

            - ``"euclidean"``, uses Exact Euclidean distance transform.
            - ``"chessboard"``, uses `chessboard` metric in chamfer type of transform.
            - ``"taxicab"``, uses `taxicab` metric in chamfer type of transform.
        spacing: spacing of pixel (or voxel) along each axis. If a sequence, must be of
            length equal to the image dimensions; if a single number, this is used for all axes.
            If ``None``, spacing of unity is used. Defaults to ``None``.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            Several input options are allowed: (1) If a single number, isotropic spacing with that value is used.
            (2) If a sequence of numbers, the length of the sequence must be equal to the image dimensions.
            (3) If ``None``, spacing of unity is used. Defaults to ``None``.

    Note:
        If seg_pred or seg_gt is all 0, may result in nan/inf distance.

    """

    if not np.any(seg_gt):
        dis = np.inf * np.ones_like(seg_gt)
    else:
        if not np.any(seg_pred):
            dis = np.inf * np.ones_like(seg_gt)
            return np.asarray(dis[seg_gt])
        if distance_metric == "euclidean":
            dis = distance_transform_edt(~seg_gt, sampling=spacing)
        elif distance_metric in {"chessboard", "taxicab"}:
            dis = distance_transform_cdt(~seg_gt, metric=distance_metric)
        else:
            raise ValueError(f"distance_metric {distance_metric} is not implemented.")

    return np.asarray(dis[seg_pred])


def is_binary_tensor(input: torch.Tensor, name: str) -> None:
    """Determines whether the input tensor is torch binary tensor or not.

    Args:
        input (torch.Tensor): tensor to validate.
        name (str): name of the tensor being checked.

    Raises:
        ValueError: if `input` is not a PyTorch Tensor.

    Note:
        A warning message is printed, if the tensor is not binary.
    """
    if not isinstance(input, torch.Tensor):
        raise ValueError(f"{name} must be of type PyTorch Tensor.")
    if not torch.all(input.byte() == input) or input.max() > 1 or input.min() < 0:
        warnings.warn(f"{name} should be a binarized tensor.")


def remap_instance_id(pred: torch.Tensor, by_size: bool = False) -> torch.Tensor:
    """
    This function is used to rename all instance id of `pred`, so that the id is
    contiguous.
    For example: all ids of the input can be [0, 1, 2] rather than [0, 2, 5].
    This function is helpful for calculating metrics like Panoptic Quality (PQ).
    The implementation refers to:

    https://github.com/vqdang/hover_net

    Args:
        pred: segmentation predictions in the form of torch tensor. Each
            value of the tensor should be an integer, and represents the prediction of its corresponding instance id.
        by_size: if True, largest instance will be assigned a smaller id.

    """
    pred_id = list(pred.unique())
    # the original implementation has the limitation that if there is no 0 in pred, error will happen
    pred_id = [i for i in pred_id if i != 0]

    if not pred_id:
        return pred
    if by_size:
        instance_size = [(pred == instance_id).sum() for instance_id in pred_id]
        pair_data = zip(pred_id, instance_size)
        pair_list = sorted(pair_data, key=lambda x: x[1], reverse=True)
        pred_id, _ = zip(*pair_list)

    new_pred = torch.zeros_like(pred, dtype=torch.int)
    for idx, instance_id in enumerate(pred_id):
        new_pred[pred == instance_id] = idx + 1
    return new_pred


def prepare_spacing(
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None,
    batch_size: int,
    img_dim: int,
) -> Sequence[None | int | float | np.ndarray | Sequence[int | float]]:
    """
    This function is used to prepare the `spacing` parameter to include batch dimension for the computation of
    surface distance, hausdorff distance or surface dice.

    An example with batch_size = 4 and img_dim = 3:
    input spacing = None -> output spacing = [None, None, None, None]
    input spacing = 0.8 -> output spacing = [0.8, 0.8, 0.8, 0.8]
    input spacing = [0.8, 0.5, 0.9] -> output spacing = [[0.8, 0.5, 0.9], [0.8, 0.5, 0.9], [0.8, 0.5, 0.9], [0.8, 0.5, 0.9]]
    input spacing = [0.8, 0.7, 1.2, 0.8] -> output spacing = [0.8, 0.7, 1.2, 0.8] (same as input)

    An example with batch_size = 3 and img_dim = 3:
    input spacing = [0.8, 0.5, 0.9] ->
    output spacing = [[0.8, 0.5, 0.9], [0.8, 0.5, 0.9], [0.8, 0.5, 0.9], [0.8, 0.5, 0.9]]

    Args:
        spacing: can be a float, a sequence of length `img_dim`, or a sequence with length `batch_size`
        that includes floats or sequences of length `img_dim`.

    Raises:
        ValueError: when `spacing` is a sequence of sequence, where the outer sequence length does not
        equal `batch_size` or inner sequence length does not equal `img_dim`.

    Returns:
        spacing: a sequence with length `batch_size` that includes integers, floats or sequences of length `img_dim`.
    """
    if spacing is None or isinstance(spacing, (int, float)):
        return list([spacing] * batch_size)
    if isinstance(spacing, (Sequence, np.ndarray)):
        if any(not isinstance(s, type(spacing[0])) for s in list(spacing)):
            raise ValueError(f"if `spacing` is a sequence, its elements should be of same type, got {spacing}.")
        if isinstance(spacing[0], (Sequence, np.ndarray)):
            if len(spacing) != batch_size:
                raise ValueError(
                    "if `spacing` is a sequence of sequences, "
                    f"the outer sequence should have same length as batch size ({batch_size}), got {spacing}."
                )
            if any(len(s) != img_dim for s in list(spacing)):
                raise ValueError(
                    "each element of `spacing` list should either have same length as"
                    f"image dim ({img_dim}), got {spacing}."
                )
            if not all(isinstance(i, (int, float)) for s in list(spacing) for i in list(s)):
                raise ValueError(
                    f"if `spacing` is a sequence of sequences or 2D np.ndarray, "
                    f"the elements should be integers or floats, got {spacing}."
                )
            return list(spacing)
        if isinstance(spacing[0], (int, float)):
            if len(spacing) != img_dim:
                raise ValueError(
                    f"if `spacing` is a sequence of numbers, "
                    f"it should have same length as image dim ({img_dim}), got {spacing}."
                )
            return [spacing for _ in range(batch_size)]  # type: ignore
        raise ValueError(f"`spacing` is a sequence of elements with unsupported type: {type(spacing[0])}")
    raise ValueError(
        f"`spacing` should either be a number, a sequence of numbers or a sequence of sequences, got {spacing}."
    )


ENCODING_KERNEL = {2: [[8, 4], [2, 1]], 3: [[[128, 64], [32, 16]], [[8, 4], [2, 1]]]}


@lru_cache(maxsize=None)
def _get_neighbour_code_to_normals_table(device=None):
    """
    returns a lookup table. For every binary neighbour code (2x2x2 neighbourhood = 8 neighbours = 8 bits = 256 codes)
    it contains the surface normals of the triangles. The length of the normal vector encodes the surfel area.
    Adapted from https://github.com/deepmind/surface-distance

    created using the marching_cube algorithm see e.g. https://en.wikipedia.org/wiki/Marching_cubes

    Args:
        device: torch device to use for the table.
    """
    zeros = [0.0, 0.0, 0.0]
    ret = [
        [zeros, zeros, zeros, zeros],
        [[0.125, 0.125, 0.125], zeros, zeros, zeros],
        [[-0.125, -0.125, 0.125], zeros, zeros, zeros],
        [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], zeros, zeros],
        [[0.125, -0.125, 0.125], zeros, zeros, zeros],
        [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], zeros, zeros],
        [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], zeros, zeros],
        [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], zeros],
        [[-0.125, 0.125, 0.125], zeros, zeros, zeros],
        [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125], zeros, zeros],
        [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], zeros, zeros],
        [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], zeros],
        [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0], zeros, zeros],
        [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], zeros],
        [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], zeros],
        [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], zeros, zeros],
        [[0.125, -0.125, -0.125], zeros, zeros, zeros],
        [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], zeros, zeros],
        [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros, zeros],
        [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], zeros],
        [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros, zeros],
        [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125], zeros],
        [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros],
        [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
        [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125], zeros, zeros],
        [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125], zeros],
        [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125], zeros],
        [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375], [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
        [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0], zeros],
        [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
        [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
        [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], zeros],
        [[0.125, -0.125, 0.125], zeros, zeros, zeros],
        [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], zeros, zeros],
        [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], zeros],
        [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], zeros],
        [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125], zeros],
        [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25], [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
        [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125], zeros],
        [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], zeros],
        [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
        [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0], zeros],
        [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
        [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375], [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
        [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], zeros],
        [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], zeros, zeros],
        [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125], zeros],
        [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25], zeros],
        [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], zeros, zeros],
        [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125], zeros],
        [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
        [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
        [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125], zeros],
        [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], zeros],
        [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
        [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
        [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], zeros],
        [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
        [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125], zeros],
        [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], zeros],
        [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], zeros, zeros],
        [[-0.125, -0.125, 0.125], zeros, zeros, zeros],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], zeros, zeros],
        [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], zeros, zeros],
        [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], zeros],
        [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], zeros, zeros],
        [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], zeros],
        [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125], zeros],
        [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25], [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
        [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125], zeros, zeros],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125], zeros],
        [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], zeros],
        [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
        [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], zeros],
        [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25], [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
        [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375], [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
        [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], zeros],
        [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], zeros, zeros],
        [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], zeros],
        [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], zeros],
        [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
        [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], zeros],
        [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5], zeros, zeros],
        [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
        [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5], zeros],
        [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125], zeros],
        [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
        [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
        [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25], zeros],
        [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375], [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
        [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], zeros],
        [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25], zeros],
        [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25], zeros, zeros],
        [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], zeros],
        [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25], zeros],
        [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
        [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125], zeros],
        [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
        [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
        [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125], zeros],
        [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], zeros],
        [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
        [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
        [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros],
        [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros],
        [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125], zeros],
        [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125], zeros, zeros],
        [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125], zeros],
        [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
        [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
        [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], zeros],
        [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0], [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
        [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5], zeros],
        [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25], zeros],
        [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], zeros, zeros],
        [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
        [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125], zeros],
        [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125], zeros],
        [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125], zeros],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], zeros, zeros],
        [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], zeros, zeros],
        [[0.125, 0.125, 0.125], zeros, zeros, zeros],
        [[0.125, 0.125, 0.125], zeros, zeros, zeros],
        [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], zeros, zeros],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], zeros, zeros],
        [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125], zeros],
        [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125], zeros],
        [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125], zeros],
        [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
        [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], zeros, zeros],
        [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25], zeros],
        [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5], zeros],
        [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0], [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
        [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], zeros],
        [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
        [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
        [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125], zeros],
        [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125], zeros, zeros],
        [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125], zeros],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros],
        [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
        [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros],
        [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
        [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
        [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], zeros],
        [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125], zeros],
        [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
        [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
        [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125], zeros],
        [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
        [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25], zeros],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], zeros],
        [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25], zeros, zeros],
        [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25], zeros],
        [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], zeros],
        [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375], [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
        [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25], zeros],
        [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
        [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
        [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125], zeros],
        [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5], zeros],
        [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
        [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5], zeros, zeros],
        [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], zeros],
        [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
        [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], zeros],
        [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], zeros],
        [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], zeros, zeros],
        [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], zeros],
        [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375], [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
        [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25], [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
        [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], zeros],
        [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
        [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], zeros],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125], zeros],
        [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125], zeros, zeros],
        [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25], [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
        [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125], zeros],
        [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], zeros],
        [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], zeros, zeros],
        [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], zeros],
        [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], zeros, zeros],
        [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], zeros, zeros],
        [[-0.125, -0.125, 0.125], zeros, zeros, zeros],
        [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], zeros, zeros],
        [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], zeros],
        [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125], zeros],
        [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
        [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], zeros],
        [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
        [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
        [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], zeros],
        [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125], zeros],
        [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
        [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
        [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125], zeros],
        [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], zeros, zeros],
        [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25], zeros],
        [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125], zeros],
        [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], zeros, zeros],
        [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], zeros],
        [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375], [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
        [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
        [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0], zeros],
        [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
        [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], zeros],
        [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125], zeros],
        [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25], [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
        [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125], zeros],
        [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], zeros],
        [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], zeros],
        [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], zeros, zeros],
        [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], zeros, zeros],
        [[0.125, -0.125, 0.125], zeros, zeros, zeros],
        [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], zeros],
        [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
        [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
        [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0], zeros],
        [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375], [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
        [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125], zeros],
        [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125], zeros],
        [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125], zeros, zeros],
        [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
        [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros],
        [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125], zeros],
        [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros, zeros],
        [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], zeros],
        [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125], zeros, zeros],
        [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], zeros, zeros],
        [[0.125, -0.125, -0.125], zeros, zeros, zeros],
        [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], zeros, zeros],
        [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], zeros],
        [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], zeros],
        [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0], zeros, zeros],
        [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], zeros],
        [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], zeros, zeros],
        [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125], zeros, zeros],
        [[-0.125, 0.125, 0.125], zeros, zeros, zeros],
        [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], zeros],
        [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], zeros, zeros],
        [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], zeros, zeros],
        [[0.125, 0.125, 0.125], zeros, zeros, zeros],
        [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], zeros, zeros],
        [[0.125, 0.125, 0.125], zeros, zeros, zeros],
        [[0.125, 0.125, 0.125], zeros, zeros, zeros],
        [zeros, zeros, zeros, zeros],
    ]
    return torch.as_tensor(ret, device=device)


def create_table_neighbour_code_to_surface_area(spacing_mm, device=None):
    """
    Returns an array mapping neighbourhood code to the surface elements area.
    Adapted from https://github.com/deepmind/surface-distance

    Note that the normals encode the initial surface area. This function computes
    the area corresponding to the given `spacing`.

    Args:
        spacing_mm: a sequence of 3 numbers. Voxel spacing along the first 3 spatial axes.
        device: device to put the table on.

    Returns:
        An array of size 256, mapping neighbourhood code to the surface area.
        ENCODING_KERNEL[3] which is the kernel used to compute the neighbourhood code.
    """
    spacing_mm = ensure_tuple_rep(spacing_mm, 3)
    # compute the area for all 256 possible surface elements given a 2x2x2 neighbourhood according to the spacing_mm
    c = _get_neighbour_code_to_normals_table(device)
    s = torch.as_tensor(
        [[[spacing_mm[1] * spacing_mm[2], spacing_mm[0] * spacing_mm[2], spacing_mm[0] * spacing_mm[1]]]],
        device=device,
        dtype=c.dtype,
    )
    norm = torch.linalg.norm(c * s, dim=-1)
    neighbour_code_to_surface_area = norm.sum(-1)
    return neighbour_code_to_surface_area, torch.as_tensor([[ENCODING_KERNEL[3]]], device=device)


def create_table_neighbour_code_to_contour_length(spacing_mm, device=None):
    """
    Returns an array mapping neighbourhood code to the contour length.
    Adapted from https://github.com/deepmind/surface-distance

    In 2D, each point has 4 neighbors. Thus, are 16 configurations. A
    configuration is encoded with '1' meaning "inside the object" and '0' "outside
    the object". For example,
    "0101" and "1010" both encode an edge along the first spatial axis with length spacing[0] mm;
    "0011" and "1100" both encode an edge along the second spatial axis with length spacing[1] mm.

    Args:
        spacing_mm: 2-element list-like structure. Pixel spacing along the 1st and 2nd spatial axes.
        device: device to put the table on.

    Returns:
        A 16-element array mapping neighbourhood code to the contour length.
        ENCODING_KERNEL[2] which is the kernel used to compute the neighbourhood code.
    """
    spacing_mm = ensure_tuple_rep(spacing_mm, 2)
    first, second = spacing_mm  # spacing along the first and second spatial dimension respectively
    diag = 0.5 * np.linalg.norm(spacing_mm)

    neighbour_code_to_contour_length = np.zeros([16], dtype=diag.dtype)
    neighbour_code_to_contour_length[int("0001", 2)] = diag
    neighbour_code_to_contour_length[int("0010", 2)] = diag
    neighbour_code_to_contour_length[int("0011", 2)] = second
    neighbour_code_to_contour_length[int("0100", 2)] = diag
    neighbour_code_to_contour_length[int("0101", 2)] = first
    neighbour_code_to_contour_length[int("0110", 2)] = 2 * diag
    neighbour_code_to_contour_length[int("0111", 2)] = diag
    neighbour_code_to_contour_length[int("1000", 2)] = diag
    neighbour_code_to_contour_length[int("1001", 2)] = 2 * diag
    neighbour_code_to_contour_length[int("1010", 2)] = first
    neighbour_code_to_contour_length[int("1011", 2)] = diag
    neighbour_code_to_contour_length[int("1100", 2)] = second
    neighbour_code_to_contour_length[int("1101", 2)] = diag
    neighbour_code_to_contour_length[int("1110", 2)] = diag
    neighbour_code_to_contour_length = convert_to_tensor(neighbour_code_to_contour_length, device=device)
    return neighbour_code_to_contour_length, torch.as_tensor([[ENCODING_KERNEL[2]]], device=device)


def get_code_to_measure_table(spacing, device=None):
    """
    returns a table mapping neighbourhood code to the surface area or contour length.

    Args:
        spacing: a sequence of 2 or 3 numbers, indicating the spacing in the spatial dimensions.
        device: device to put the table on.
    """
    spatial_dims = len(spacing)
    spacing = ensure_tuple_rep(spacing, look_up_option(spatial_dims, (2, 3)))
    if spatial_dims == 2:
        return create_table_neighbour_code_to_contour_length(spacing, device)
    return create_table_neighbour_code_to_surface_area(spacing, device)
