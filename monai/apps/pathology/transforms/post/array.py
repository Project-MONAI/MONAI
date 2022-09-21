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

from typing import Optional, Sequence, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import get_track_meta

from monai.transforms.post.array import GetInstanceLevelSegMap, Activations, AsDiscrete
from monai.transforms.croppad.array import BoundingRect
from monai.transforms.transform import Transform
from monai.utils import convert_to_numpy, convert_to_tensor, optional_import
from monai.utils.enums import TransformBackends

find_contours, has_findcontours = optional_import("skimage.measure", name="find_contours")
moments, has_moments = optional_import("skimage.measure", name="moments")

__all__ = ["PostProcessHoVerNetOutput"]


def _coords_to_pixel(current, previous):
    """For contour coordinate generation.
    Given the previous and current border positions,
    returns the int pixel that marks the extremity
    of the segmented pixels
    """

    p_delta = (current[0] - previous[0], current[1] - previous[1])

    if p_delta == (0.0, 1.0) or p_delta == (0.5, 0.5) or p_delta == (1.0, 0.0):
        row = int(current[0] + 0.5)
        col = int(current[1])
    elif p_delta == (0.0, -1.0) or p_delta == (0.5, -0.5):
        row = int(current[0])
        col = int(current[1])
    elif p_delta == (-1, 0.0) or p_delta == (-0.5, -0.5):
        row = int(current[0])
        col = int(current[1] + 0.5)
    elif p_delta == (-0.5, 0.5):
        row = int(current[0] + 0.5)
        col = int(current[1] + 0.5)

    return row, col


def _dist_from_topleft(sequence, h, w):
    """For contour coordinate generation.
    Each sequence of cordinates describes a boundary between
    foreground and background starting and ending at two sides
    of the bounding box. To order the sequences correctly,
    we compute the distannce from the topleft of the bounding box
    around the perimeter in a clock-wise direction.
     Args:
         sequence: list of border points
         h: height of the bounding box
         w: width of the bounding box
     Returns:
         distance: the distance round the perimeter of the bounding
             box from the top-left origin
    """

    first = sequence[0]
    if first[0] == 0:
        distance = first[1]
    elif first[1] == w - 1:
        distance = w + first[0]
    elif first[0] == h - 1:
        distance = 2 * w + h - first[1]
    else:
        distance = 2 * (w + h) - first[0]

    return distance


def _sp_contours_to_cv(contours, h, w):
    """Converts Scipy-style contours to a more succinct version
       which only includes the pixels to which lines need to
       be drawn (i.e. not the intervening pixels along each line).
    Args:
        contours: scipy-style clockwise line segments, with line separating foreground/background
        h: Height of bounding box - used to detect direction of line segment
        w: Width of bounding box - used to detect direction of line segment
    Returns:
        pixels: the pixels that need to be joined by straight lines to
                describe the outmost pixels of the foreground similar to
                OpenCV's cv.CHAIN_APPROX_SIMPLE (anti-clockwise)
    """
    pixels = None
    sequences = []
    corners = [False, False, False, False]

    for group in contours:
        sequence = []
        last_added = None
        prev = None
        corner = -1

        for i, coord in enumerate(group):
            if i == 0:
                if coord[0] == 0.0:
                    # originating from the top, so must be heading south east
                    corner = 1
                    pixel = (0, int(coord[1] - 0.5))
                    if pixel[1] == w - 1:
                        corners[1] = True
                    elif pixel[1] == 0.0:
                        corners[0] = True
                elif coord[1] == 0.0:
                    corner = 0
                    # originating from the left, so must be heading north east
                    pixel = (int(coord[0] + 0.5), 0)
                elif coord[0] == h - 1:
                    corner = 3
                    # originating from the bottom, so must be heading north west
                    pixel = (int(coord[0]), int(coord[1] + 0.5))
                    if pixel[1] == w - 1:
                        corners[2] = True
                elif coord[1] == w - 1:
                    corner = 2
                    # originating from the right, so must be heading south west
                    pixel = (int(coord[0] - 0.5), int(coord[1]))

                sequence.append(pixel)
                last_added = pixel
            elif i == len(group) - 1:
                # add this point
                pixel = _coords_to_pixel(coord, prev)
                if pixel != last_added:
                    sequence.append(pixel)
                    last_added = pixel
            elif np.any(coord - prev != group[i + 1] - coord):
                pixel = _coords_to_pixel(coord, prev)
                if pixel != last_added:
                    sequence.append(pixel)
                    last_added = pixel

            # flag whether each corner has been crossed
            if i == len(group) - 1:
                if corner == 0:
                    if coord[0] == 0:
                        corners[corner] = True
                elif corner == 1:
                    if coord[1] == w - 1:
                        corners[corner] = True
                elif corner == 2:
                    if coord[0] == h - 1:
                        corners[corner] = True
                elif corner == 3:
                    if coord[1] == 0.0:
                        corners[corner] = True

            prev = coord

        dist = _dist_from_topleft(sequence, h, w)

        sequences.append({"distance": dist, "sequence": sequence})

    # check whether we need to insert any missing corners
    if corners[0] is False:
        sequences.append({"distance": 0, "sequence": [(0, 0)]})
    if corners[1] is False:
        sequences.append({"distance": w, "sequence": [(0, w - 1)]})
    if corners[2] is False:
        sequences.append({"distance": w + h, "sequence": [(h - 1, w - 1)]})
    if corners[3] is False:
        sequences.append({"distance": 2 * w + h, "sequence": [(h - 1, 0)]})

    # now, join the sequences into a single contour
    # starting at top left and rotating clockwise
    sequences.sort(key=lambda x: x.get("distance"))

    last = (-1, -1)
    for sequence in sequences:
        if sequence["sequence"][0] == last:
            pixels.pop()

        if pixels:
            pixels = [*pixels, *sequence["sequence"]]
        else:
            pixels = sequence["sequence"]

        last = pixels[-1]

    if pixels[0] == last:
        pixels.pop(0)

    if pixels[0] == (0, 0):
        pixels.append(pixels.pop(0))

    pixels = np.array(pixels).astype("int32")
    pixels = np.flip(pixels)

    return pixels


class PostProcessHoVerNetOutput(Transform):
    """Post processing script for image tiles.
    Args:
        output_classes: number of types considered at output of NC branch.
        return_centroids: whether to generate coords for each nucleus instance.
            Defaults to True.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        output_classes: Optional[int] = None,
        return_centroids: bool = True,
        threshold_pred: float = 0.5,
        threshold_overall: float = 0.4,
        min_size: int = 10,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.4,
        kernel_size: int = 17,
        radius: int = 2,
    ) -> None:
        self.output_classes = output_classes
        self.return_centroids = return_centroids

        self.get_bbox = BoundingRect()
        self.get_instance_level_seg_map = GetInstanceLevelSegMap(
            threshold_pred=threshold_pred,
            threshold_overall=threshold_overall,
            min_size=min_size,
            sigma=sigma,
            kernel_size=kernel_size,
            radius=radius,
        )

    def __call__(self, NP_pred: NdarrayOrTensor, HV_pred: NdarrayOrTensor, NC_pred: Optional[NdarrayOrTensor] = None):
        """
        Args:
            NP_pred: segmentation branch output with shape [1, W, H, [D]].
            HV_pred: hover map branch output with shape [2, W, H, [D]].
            NC_pred: classification branch output with shape [1, W, H, [D]]. Defaults to None.
        Returns:
            pred_inst: pixel-wise nuclear instance segmentation prediction.
            inst_info_dict: a instance-level information dictionary containing bounding_box, centroid and contour.
                If output_classes is not None, the dictionary will also contain pixel-wise nuclear type prediction.
        """
        NP_pred = convert_to_tensor(NP_pred, track_meta=get_track_meta())
        HV_pred = convert_to_tensor(HV_pred, track_meta=get_track_meta())
        if NC_pred is not None:
            NC_pred = AsDiscrete(argmax=True)(Activations(softmax=True)(NC_pred))
            NC_pred = convert_to_tensor(NC_pred, track_meta=get_track_meta())

        pred_inst = self.get_instance_level_seg_map(NP_pred, HV_pred)

        inst_info_dict = None
        if self.return_centroids or self.output_classes is not None:
            inst_id_list = torch.unique(pred_inst)[1:]  # exlcude background
            inst_info_dict = {}
            for inst_id in inst_id_list:
                inst_map = pred_inst == inst_id
                inst_bbox = self.get_bbox(inst_map)
                inst_map = inst_map[inst_bbox[0][0] : inst_bbox[0][1], inst_bbox[1][0] : inst_bbox[1][1]]
                inst_map_np = convert_to_numpy(inst_map.squeeze(), dtype=np.uint8)  # squeeze remove channel dim
                inst_moment = moments(inst_map_np, order=3)

                inst_contour_cv = find_contours(inst_map_np, 0.5)
                inst_contour = _sp_contours_to_cv(inst_contour_cv, inst_map.shape[0], inst_map.shape[1])

                # < 3 points dont make a contour, so skip, likely artifact too
                # as the contours obtained via approximation => too small or sthg
                if inst_contour.shape[0] < 3:
                    continue
                if len(inst_contour.shape) != 2:
                    continue  # ! check for tricky shape

                inst_centroid = [(inst_moment[0, 1] / inst_moment[0, 0]), (inst_moment[1, 0] / inst_moment[0, 0])]
                inst_centroid = np.array(inst_centroid)
                inst_contour[:, 0] += inst_bbox[0][1]  # X
                inst_contour[:, 1] += inst_bbox[0][0]  # Y
                inst_centroid[0] += inst_bbox[0][1]  # X why [0][1] represent x??
                inst_centroid[1] += inst_bbox[0][0]  # Y
                inst_info_dict[inst_id] = {  # inst_id should start at 1
                    "bounding_box": inst_bbox,
                    "centroid": inst_centroid,
                    "contour": inst_contour,
                    "type_probability": None,
                    "type": None,
                }
        if self.output_classes is not None:
            for inst_id in list(inst_info_dict.keys()):
                rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bounding_box"]).flatten()
                inst_map_crop = pred_inst[:, rmin:rmax, cmin:cmax]
                inst_type_crop = NC_pred[:, rmin:rmax, cmin:cmax]
                inst_map_crop = inst_map_crop == inst_id
                inst_type = inst_type_crop[inst_map_crop]
                type_list, type_pixels = torch.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                inst_type = type_list[0][0]
                if inst_type == 0:  # ! pick the 2nd most dominant if exist
                    if len(type_list) > 1:
                        inst_type = type_list[1][0]
                type_dict = {v[0]: v[1] for v in type_list}
                type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
                inst_info_dict[inst_id]["type"] = int(inst_type)
                inst_info_dict[inst_id]["type_probability"] = float(type_prob)

        return (pred_inst, inst_info_dict)
