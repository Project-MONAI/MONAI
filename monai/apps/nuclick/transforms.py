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

import math
import random
from enum import Enum
from typing import Tuple, Union

import numpy as np

from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable, SpatialPad
from monai.utils import optional_import

measure, _ = optional_import("skimage.measure")
morphology, _ = optional_import("skimage.morphology")


class NuclickKeys(Enum):
    """
    Keys for nuclick transforms.
    """

    IMAGE = "image"
    LABEL = "label"
    OTHERS = "others"  # key of other labels from the binary mask which are not being used for training
    FOREGROUND = "foreground"

    CENTROID = "centroid"  # key where the centroid values are stored
    MASK_VALUE = "mask_value"
    LOCATION = "location"

    NUC_POINTS = "nuc_points"
    BOUNDING_BOXES = "bounding_boxes"
    IMG_HEIGHT = "img_height"
    IMG_WIDTH = "img_width"


class FlattenLabeld(MapTransform):
    """
    FlattenLabeld creates labels per closed object contour (defined by a connectivity). For e.g if there are
    12 small regions of 1's it will delineate them into 12 different label classes

    Args:
        connectivity: Max no. of orthogonal hops to consider a pixel/voxel as a neighbor. Refer skimage.measure.label
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(self, keys: KeysCollection, connectivity: int = 1, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.connectivity = connectivity

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = measure.label(d[key], connectivity=self.connectivity).astype(np.uint8)
        return d


class ExtractPatchd(MapTransform):
    """
    Extracts a patch from the given image and label, however it is based on the centroid location.
    The centroid location is a 2D coordinate (H, W). The extracted patch is extracted around the centroid,
    if the centroid is towards the edge, the centroid will not be the center of the image as the patch will be
    extracted from the edges onwards

    Args:
        keys: image, label
        centroid_key: key where the centroid values are stored, defaults to ``"centroid"``
        patch_size: size of the extracted patch
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        centroid_key: str = NuclickKeys.CENTROID.value,
        patch_size: Union[Tuple[int, int], int] = 128,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.centroid_key = centroid_key
        self.patch_size = patch_size

    def __call__(self, data):
        d = dict(data)

        centroid = d[self.centroid_key]  # create mask based on centroid (select nuclei based on centroid)
        roi_size = (self.patch_size, self.patch_size)

        for key in self.keys:
            img = d[key]
            x_start, x_end, y_start, y_end = self.bbox(self.patch_size, centroid, img.shape[-2:])
            cropped = img[:, x_start:x_end, y_start:y_end]
            d[key] = SpatialPad(spatial_size=roi_size)(cropped)
        return d

    def bbox(self, patch_size, centroid, size):
        x, y = centroid
        m, n = size

        x_start = int(max(x - patch_size / 2, 0))
        y_start = int(max(y - patch_size / 2, 0))
        x_end = x_start + patch_size
        y_end = y_start + patch_size
        if x_end > m:
            x_end = m
            x_start = m - patch_size
        if y_end > n:
            y_end = n
            y_start = n - patch_size
        return x_start, x_end, y_start, y_end


class SplitLabeld(MapTransform):
    """
    Extracts a single label from all the given classes, the single label is defined by mask_value, the remaining
    labels are kept in others

    Args:
        label: label source, defaults to ``"label"``
        others: other labels storage key, defaults to ``"others"``
        mask_value: the mask_value that will be kept for binarization of the label, defaults to ``"mask_value"``
        min_area: The smallest allowable object size.
    """

    def __init__(
        self,
        label: str = NuclickKeys.LABEL.value,
        others: str = NuclickKeys.OTHERS.value,
        mask_value: str = NuclickKeys.MASK_VALUE.value,
        min_area: int = 5,
    ):

        self.label = label
        self.others = others
        self.mask_value = mask_value
        self.min_area = min_area

    def __call__(self, data):
        d = dict(data)
        label = d[self.label]
        mask_value = d[self.mask_value]
        mask = np.uint8(label == mask_value)
        others = (1 - mask) * label
        others = self._mask_relabeling(others[0], min_area=self.min_area)[np.newaxis]
        d[self.label] = mask
        d[self.others] = others
        return d

    def _mask_relabeling(self, mask, min_area=5):
        res = np.zeros_like(mask)
        for l in np.unique(mask):
            if l == 0:
                continue

            m = measure.label(mask == l, connectivity=1)
            for stat in measure.regionprops(m):
                if stat.area > min_area:
                    res[stat.coords[:, 0], stat.coords[:, 1]] = l
        return res


class FilterImaged(MapTransform):
    """
    Filters Green and Gray channel of the image using an allowable object size, this pre-processing transform
    is specific towards NuClick training process. More details can be referred in this paper Koohbanani,
    Navid Alemi, et al. "NuClick: a deep learning framework for interactive segmentation of microscopic images."
    Medical Image Analysis 65 (2020): 101771.

    Args:
        min_size: The smallest allowable object size
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(self, keys: KeysCollection, min_size: int = 500, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.min_size = min_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            d[key] = self.filter(img)
        return d

    def filter(self, rgb):
        mask_not_green = self.filter_green_channel(rgb)
        mask_not_gray = self.filter_grays(rgb)
        mask_gray_green = mask_not_gray & mask_not_green
        mask = (
            self.filter_remove_small_objects(mask_gray_green, min_size=self.min_size)
            if self.min_size
            else mask_gray_green
        )

        return rgb * np.dstack([mask, mask, mask])

    def filter_green_channel(
        self, img_np, green_thresh=200, avoid_overmask=True, overmask_thresh=90, output_type="bool"
    ):
        g = img_np[:, :, 1]
        gr_ch_mask = (g < green_thresh) & (g > 0)
        mask_percentage = self.mask_percent(gr_ch_mask)
        if (mask_percentage >= overmask_thresh) and (green_thresh < 255) and (avoid_overmask is True):
            new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
            gr_ch_mask = self.filter_green_channel(
                img_np, new_green_thresh, avoid_overmask, overmask_thresh, output_type
            )
        return gr_ch_mask

    def filter_grays(self, rgb, tolerance=15):
        rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
        rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
        gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
        return ~(rg_diff & rb_diff & gb_diff)

    def mask_percent(self, img_np):
        if (len(img_np.shape) == 3) and (img_np.shape[2] == 3):
            np_sum = img_np[:, :, 0] + img_np[:, :, 1] + img_np[:, :, 2]
            mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
        else:
            mask_percentage = 100 - np.count_nonzero(img_np) / img_np.size * 100
        return mask_percentage

    def filter_remove_small_objects(self, img_np, min_size=3000, avoid_overmask=True, overmask_thresh=95):
        rem_sm = morphology.remove_small_objects(img_np.astype(bool), min_size=min_size)
        mask_percentage = self.mask_percent(rem_sm)
        if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
            new_min_size = round(min_size / 2)
            rem_sm = self.filter_remove_small_objects(img_np, new_min_size, avoid_overmask, overmask_thresh)
        return rem_sm


class AddPointGuidanceSignald(Randomizable, MapTransform):
    """
    Adds Guidance Signal to the input image

    Args:
        image: key of source image, defaults to ``"image"``
        label: key of source label, defaults to ``"label"``
        others: source others (other labels from the binary mask which are not being used for training)
            defaults to ``"others"``
        drop_rate: probability of dropping the signal, defaults to ``0.5``
        jitter_range: noise added to the points in the point mask for exclusion mask, defaults to ``3``
    """

    def __init__(
        self,
        image: str = NuclickKeys.IMAGE.value,
        label: str = NuclickKeys.LABEL.value,
        others: str = NuclickKeys.OTHERS.value,
        drop_rate: float = 0.5,
        jitter_range: int = 3,
    ):
        MapTransform.__init__(self, image)

        self.image = image
        self.label = label
        self.others = others
        self.drop_rate = drop_rate
        self.jitter_range = jitter_range

    def __call__(self, data):
        d = dict(data)

        image = d[self.image]
        mask = d[self.label]
        others = d[self.others]

        inc_sig = self.inclusion_map(mask[0])
        exc_sig = self.exclusion_map(others[0], drop_rate=self.drop_rate, jitter_range=self.jitter_range)

        image = np.concatenate((image, inc_sig[np.newaxis, ...], exc_sig[np.newaxis, ...]), axis=0)
        d[self.image] = image
        return d

    def inclusion_map(self, mask):
        point_mask = np.zeros_like(mask)
        indices = np.argwhere(mask > 0)
        if len(indices) > 0:
            idx = np.random.randint(0, len(indices))
            point_mask[indices[idx, 0], indices[idx, 1]] = 1

        return point_mask

    def exclusion_map(self, others, jitter_range=3, drop_rate=0.5):
        point_mask = np.zeros_like(others)
        if drop_rate == 1.0:
            return point_mask

        max_x = point_mask.shape[0] - 1
        max_y = point_mask.shape[1] - 1
        stats = measure.regionprops(others)
        for stat in stats:
            x, y = stat.centroid
            if np.random.choice([True, False], p=[drop_rate, 1 - drop_rate]):
                continue

            # random jitter
            x = int(math.floor(x)) + random.randint(a=-jitter_range, b=jitter_range)
            y = int(math.floor(y)) + random.randint(a=-jitter_range, b=jitter_range)
            x = min(max(0, x), max_x)
            y = min(max(0, y), max_y)
            point_mask[x, y] = 1

        return point_mask


class AddClickSignalsd(MapTransform):
    """
    Adds Click Signal to the input image

    Args:
        image: source image, defaults to ``"image"``
        foreground: 2D click indices as list, defaults to ``"foreground"``
        bb_size: single integer size, defines a bounding box like (bb_size, bb_size)
    """

    def __init__(
        self, image: str = NuclickKeys.IMAGE.value, foreground: str = NuclickKeys.FOREGROUND.value, bb_size: int = 128
    ):
        self.image = image
        self.foreground = foreground
        self.bb_size = bb_size

    def __call__(self, data):
        d = dict(data)

        location = d.get(NuclickKeys.LOCATION.value, (0, 0))
        tx, ty = location[0], location[1]
        pos = d.get(self.foreground)
        pos = (np.array(pos) - (tx, ty)).astype(int).tolist() if pos else []

        cx = [xy[0] for xy in pos]
        cy = [xy[1] for xy in pos]

        img = d[self.image].astype(np.uint8)
        img_width = img.shape[-1]
        img_height = img.shape[-2]

        click_map, bounding_boxes = self.get_clickmap_boundingbox(
            cx=cx, cy=cy, m=img_height, n=img_width, bb=self.bb_size
        )

        patches, nuc_points, other_points = self.get_patches_and_signals(
            img=img,
            click_map=click_map,
            bounding_boxes=bounding_boxes,
            cx=cx,
            cy=cy,
            m=img_height,
            n=img_width,
            bb=self.bb_size,
        )
        patches = patches / 255

        d[NuclickKeys.BOUNDING_BOXES.value] = bounding_boxes
        d[NuclickKeys.IMG_WIDTH.value] = img_width
        d[NuclickKeys.IMG_HEIGHT.value] = img_height
        d[NuclickKeys.NUC_POINTS.value] = nuc_points

        d[self.image] = np.concatenate((patches, nuc_points, other_points), axis=1).astype(dtype=np.float32)
        return d

    def get_clickmap_boundingbox(self, cx, cy, m, n, bb=128):
        click_map = np.zeros((m, n), dtype=np.uint8)

        x_del_indices = {i for i in range(len(cx)) if cx[i] >= n or cx[i] < 0}
        y_del_indices = {i for i in range(len(cy)) if cy[i] >= m or cy[i] < 0}
        del_indices = list(x_del_indices.union(y_del_indices))
        cx = np.delete(cx, del_indices)
        cy = np.delete(cy, del_indices)

        click_map[cy, cx] = 1
        bounding_boxes = []
        for i in range(len(cx)):
            x_start = cx[i] - bb // 2
            y_start = cy[i] - bb // 2
            if x_start < 0:
                x_start = 0
            if y_start < 0:
                y_start = 0
            x_end = x_start + bb - 1
            y_end = y_start + bb - 1
            if x_end > n - 1:
                x_end = n - 1
                x_start = x_end - bb + 1
            if y_end > m - 1:
                y_end = m - 1
                y_start = y_end - bb + 1
            bounding_boxes.append([x_start, y_start, x_end, y_end])
        return click_map, bounding_boxes

    def get_patches_and_signals(self, img, click_map, bounding_boxes, cx, cy, m, n, bb=128):

        total = len(bounding_boxes)
        img = np.array([img])
        click_map = np.array([click_map])
        click_map = click_map[:, np.newaxis, ...]

        patches = np.ndarray((total, 3, bb, bb), dtype=np.uint8)
        nuc_points = np.ndarray((total, 1, bb, bb), dtype=np.uint8)
        other_points = np.ndarray((total, 1, bb, bb), dtype=np.uint8)

        x_del_indices = {i for i in range(len(cx)) if cx[i] >= n or cx[i] < 0}
        y_del_indices = {i for i in range(len(cy)) if cy[i] >= m or cy[i] < 0}
        del_indices = list(x_del_indices.union(y_del_indices))
        cx = np.delete(cx, del_indices)
        cy = np.delete(cy, del_indices)

        for i in range(len(bounding_boxes)):
            bounding_box = bounding_boxes[i]
            x_start = bounding_box[0]
            y_start = bounding_box[1]
            x_end = bounding_box[2]
            y_end = bounding_box[3]

            patches[i] = img[0, :, y_start : y_end + 1, x_start : x_end + 1]

            this_click_map = np.zeros((1, 1, m, n), dtype=np.uint8)
            this_click_map[0, 0, cy[i], cx[i]] = 1

            others_click_map = np.uint8((click_map - this_click_map) > 0)

            nuc_points[i] = this_click_map[0, :, y_start : y_end + 1, x_start : x_end + 1]
            other_points[i] = others_click_map[0, :, y_start : y_end + 1, x_start : x_end + 1]

        return patches, nuc_points, other_points


class PostFilterLabeld(MapTransform):
    """
    Performs Filtering of Labels on the predicted probability map

    Args:
        thresh: probability threshold for classifying a pixel as a mask
        min_size: min_size objects that will be removed from the image, refer skimage remove_small_objects
        min_hole: min_hole that will be removed from the image, refer skimage remove_small_holes
        do_reconstruction: Boolean Flag, Perform a morphological reconstruction of an image, refer skimage
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        nuc_points: str = NuclickKeys.NUC_POINTS.value,
        bounding_boxes: str = NuclickKeys.BOUNDING_BOXES.value,
        img_height: str = NuclickKeys.IMG_HEIGHT.value,
        img_width: str = NuclickKeys.IMG_WIDTH.value,
        thresh: float = 0.33,
        min_size: int = 10,
        min_hole: int = 30,
        do_reconstruction: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.nuc_points = nuc_points
        self.bounding_boxes = bounding_boxes
        self.img_height = img_height
        self.img_width = img_width

        self.thresh = thresh
        self.min_size = min_size
        self.min_hole = min_hole
        self.do_reconstruction = do_reconstruction

    def __call__(self, data):
        d = dict(data)

        nuc_points = d[self.nuc_points]
        bounding_boxes = d[self.bounding_boxes]
        img_height = d[self.img_height]
        img_width = d[self.img_width]

        for key in self.keys:
            label = d[key].astype(np.uint8)
            masks = self.post_processing(
                label,
                thresh=self.thresh,
                min_size=self.min_size,
                min_hole=self.min_hole,
                do_reconstruction=self.do_reconstruction,
                nuc_points=nuc_points,
            )

            d[key] = self.gen_instance_map(masks, bounding_boxes, img_height, img_width).astype(np.uint8)
        return d

    def post_processing(self, preds, thresh=0.33, min_size=10, min_hole=30, do_reconstruction=False, nuc_points=None):
        masks = preds > thresh
        masks = morphology.remove_small_objects(masks, min_size=min_size)
        masks = morphology.remove_small_holes(masks, area_threshold=min_hole)
        if do_reconstruction:
            for i in range(len(masks)):
                this_mask = masks[i]
                this_marker = nuc_points[i, 0, :, :] > 0

                try:
                    this_mask = morphology.reconstruction(this_marker, this_mask, footprint=morphology.disk(1))
                    masks[i] = np.array([this_mask])
                except BaseException:
                    print("Nuclei reconstruction error #" + str(i))
        return masks

    def gen_instance_map(self, masks, bounding_boxes, m, n, flatten=True):
        instance_map = np.zeros((m, n), dtype=np.uint16)
        for i in range(len(masks)):
            this_bb = bounding_boxes[i]
            this_mask_pos = np.argwhere(masks[i] > 0)
            this_mask_pos[:, 0] = this_mask_pos[:, 0] + this_bb[1]
            this_mask_pos[:, 1] = this_mask_pos[:, 1] + this_bb[0]
            instance_map[this_mask_pos[:, 0], this_mask_pos[:, 1]] = 1 if flatten else i + 1
        return instance_map
