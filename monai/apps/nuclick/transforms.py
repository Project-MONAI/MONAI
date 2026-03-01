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
from typing import Any

import numpy as np
import torch

from monai.config import KeysCollection, NdarrayOrTensor
from monai.networks.layers import GaussianFilter
from monai.transforms import MapTransform, Randomizable, SpatialPad
from monai.utils import StrEnum, convert_to_numpy, optional_import

measure, _ = optional_import("skimage.measure")
morphology, _ = optional_import("skimage.morphology")
distance_transform_cdt, _ = optional_import("scipy.ndimage", name="distance_transform_cdt")


class NuclickKeys(StrEnum):
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
    PRED_CLASSES = "pred_classes"


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
            img = convert_to_numpy(d[key]) if isinstance(d[key], torch.Tensor) else d[key]
            d[key] = measure.label(img, connectivity=self.connectivity).astype(np.uint8)
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
        pad_kwargs: other arguments for the SpatialPad transform
    """

    def __init__(
        self,
        keys: KeysCollection,
        centroid_key: str = NuclickKeys.CENTROID,
        patch_size: tuple[int, int] | int = 128,
        allow_missing_keys: bool = False,
        **kwargs: Any,
    ):
        super().__init__(keys, allow_missing_keys)
        self.centroid_key = centroid_key
        self.patch_size = patch_size
        self.kwargs = kwargs

    def __call__(self, data):
        d = dict(data)

        centroid = d[self.centroid_key]  # create mask based on centroid (select nuclei based on centroid)
        roi_size = (self.patch_size, self.patch_size)

        for key in self.keys:
            img = d[key]
            x_start, x_end, y_start, y_end = self.bbox(self.patch_size, centroid, img.shape[-2:])
            cropped = img[:, x_start:x_end, y_start:y_end]
            d[key] = SpatialPad(spatial_size=roi_size, **self.kwargs)(cropped)
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
        label: key of the label source
        others: other labels storage key, defaults to ``"others"``
        mask_value: the mask_value that will be kept for binarization of the label, defaults to ``"mask_value"``
        min_area: The smallest allowable object size.
        others_value: Value/class for other nuclei;  Use this to separate core nuclei vs others.
        to_binary_mask: Convert mask to binary;  Set it false to restore original class values
    """

    def __init__(
        self,
        keys: KeysCollection,
        others: str = NuclickKeys.OTHERS,
        mask_value: str | None = NuclickKeys.MASK_VALUE,
        min_area: int = 5,
        others_value: int = 0,
        to_binary_mask: bool = True,
    ):
        super().__init__(keys, allow_missing_keys=False)
        self.others = others
        self.mask_value = mask_value
        self.min_area = min_area
        self.others_value = others_value
        self.to_binary_mask = to_binary_mask

    def __call__(self, data):
        d = dict(data)

        if len(self.keys) > 1:
            print("Only 'label' key is supported, more than 1 key was found")
            return None

        for key in self.keys:
            label = d[key] if isinstance(d[key], torch.Tensor) else torch.from_numpy(d[key])

            mask = torch.clone(label)
            if self.mask_value:
                mask_value = d[self.mask_value]
                mask[label != mask_value] = 0
            else:
                mask[label >= self.others_value] = 0
                mask_value = int(torch.max(mask))

            if self.to_binary_mask:
                mask[mask > 0] = 1

            others = torch.clone(label)
            others[label == mask_value] = 0
            others[others > 0] = 1
            if torch.count_nonzero(others):
                others = measure.label(convert_to_numpy(others)[0], connectivity=1)
                others = torch.from_numpy(others)[None]

            label = mask.type(torch.uint8) if isinstance(mask, torch.Tensor) else mask
            others = others.type(torch.uint8) if isinstance(others, torch.Tensor) else others

            d[key] = label if isinstance(d[key], torch.Tensor) else convert_to_numpy(label)
            d[self.others] = others if isinstance(d[key], torch.Tensor) else convert_to_numpy(others)

        return d


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
            img = convert_to_numpy(d[key]) if isinstance(d[key], torch.Tensor) else d[key]
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
        gaussian: add gaussian
        sigma: sigma value for gaussian
        truncated: spreads how many stds for gaussian
        add_exclusion_map: add exclusion map/signal
    """

    def __init__(
        self,
        image: str = NuclickKeys.IMAGE,
        label: str = NuclickKeys.LABEL,
        others: str = NuclickKeys.OTHERS,
        drop_rate: float = 0.5,
        jitter_range: int = 0,
        gaussian: bool = False,
        sigma: float = 1.0,
        truncated: float = 2.0,
        add_exclusion_map: bool = True,
        use_distance: bool = False,
    ):
        MapTransform.__init__(self, image)

        self.image = image
        self.label = label
        self.others = others
        self.drop_rate = drop_rate
        self.jitter_range = jitter_range
        self.gaussian = gaussian
        self.sigma = sigma
        self.truncated = truncated
        self.add_exclusion_map = add_exclusion_map
        self.use_distance = use_distance

    def __call__(self, data):
        d = dict(data)

        image = d[self.image] if isinstance(d[self.image], torch.Tensor) else torch.from_numpy(d[self.image])
        mask = d[self.label] if isinstance(d[self.label], torch.Tensor) else torch.from_numpy(d[self.label])

        inc_sig = self.inclusion_map(mask[0], dtype=image.dtype)
        inc_sig = self._apply_gaussian(inc_sig)
        if self.add_exclusion_map:
            others = d[self.others] if isinstance(d[self.others], torch.Tensor) else torch.from_numpy(d[self.others])
            exc_sig = self.exclusion_map(
                others[0], dtype=image.dtype, drop_rate=self.drop_rate, jitter_range=self.jitter_range
            )
            exc_sig = self._apply_gaussian(exc_sig)
            image = torch.cat((image, inc_sig[None], exc_sig[None]), dim=0)
        else:
            image = torch.cat((image, inc_sig[None]), dim=0)

        d[self.image] = image if isinstance(d[self.image], torch.Tensor) else convert_to_numpy(image)
        return d

    def _apply_gaussian(self, t):
        if not self.gaussian or torch.count_nonzero(t) == 0:
            return t
        x = GaussianFilter(spatial_dims=2, truncated=self.truncated, sigma=self.sigma)(t.unsqueeze(0).unsqueeze(0))
        return x.squeeze(0).squeeze(0)

    def _seed_point(self, label):
        if distance_transform_cdt is None or not self.use_distance:
            indices: NdarrayOrTensor
            if hasattr(torch, "argwhere"):
                indices = torch.argwhere(label > 0)
            else:
                indices = np.argwhere(convert_to_numpy(label) > 0)

            if len(indices) > 0:
                index = self.R.randint(0, len(indices))
                return indices[index, 0], indices[index, 1]
            return None

        distance = distance_transform_cdt(label).flatten()
        probability = np.exp(distance) - 1.0

        idx = np.where(label.flatten() > 0)[0]
        seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
        g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
        return g[-2], g[-1]

    def inclusion_map(self, mask, dtype):
        point_mask = torch.zeros_like(mask, dtype=dtype)
        pt = self._seed_point(mask)
        if pt is not None:
            point_mask[pt[0], pt[1]] = 1

        return point_mask

    def exclusion_map(self, others, dtype, jitter_range, drop_rate):
        point_mask = torch.zeros_like(others, dtype=dtype)
        if np.random.choice([True, False], p=[drop_rate, 1 - drop_rate]):
            return point_mask

        max_x = point_mask.shape[0] - 1
        max_y = point_mask.shape[1] - 1
        stats = measure.regionprops(convert_to_numpy(others))
        for stat in stats:
            if np.random.choice([True, False], p=[drop_rate, 1 - drop_rate]):
                continue

            # random jitter
            x, y = stat.centroid
            x = int(math.floor(x))
            y = int(math.floor(y))
            if jitter_range:
                x = x + self.R.randint(low=-jitter_range, high=jitter_range)
                y = y + self.R.randint(low=-jitter_range, high=jitter_range)
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
        gaussian: add gaussian
        sigma: sigma value for gaussian
        truncated: spreads how many stds for gaussian
        add_exclusion_map: add exclusion map/signal
    """

    def __init__(
        self,
        image: str = NuclickKeys.IMAGE,
        foreground: str = NuclickKeys.FOREGROUND,
        bb_size: int = 128,
        gaussian: bool = False,
        sigma: float = 1.0,
        truncated: float = 2.0,
        add_exclusion_map: bool = True,
    ):
        self.image = image
        self.foreground = foreground
        self.bb_size = bb_size
        self.gaussian = gaussian
        self.sigma = sigma
        self.truncated = truncated
        self.add_exclusion_map = add_exclusion_map

    def __call__(self, data):
        d = dict(data)

        img = d[self.image] if isinstance(d[self.image], torch.Tensor) else torch.from_numpy(d[self.image])
        x = img.shape[-2]
        y = img.shape[-1]

        location = d.get(NuclickKeys.LOCATION.value, (0, 0))
        tx, ty = location[0], location[1]
        pos = d.get(self.foreground)
        pos = (np.array(pos) - (tx, ty)).astype(int).tolist() if pos else []

        cx = [xy[0] for xy in pos]
        cy = [xy[1] for xy in pos]

        click_map, bounding_boxes = self.get_clickmap_boundingbox(img, cx=cx, cy=cy, x=x, y=y, bb=self.bb_size)
        if not bounding_boxes:
            raise ValueError("Failed to create patches from given click points")

        patches = self.get_patches_and_signals(
            img=img, click_map=click_map, bounding_boxes=bounding_boxes, cx=cx, cy=cy, x=x, y=y
        )

        d[NuclickKeys.BOUNDING_BOXES.value] = bounding_boxes
        d[NuclickKeys.IMG_WIDTH.value] = x
        d[NuclickKeys.IMG_HEIGHT.value] = y

        d[self.image] = patches if isinstance(d[self.image], torch.Tensor) else convert_to_numpy(patches)
        return d

    def get_clickmap_boundingbox(self, img, cx, cy, x, y, bb=128):
        click_map = torch.zeros_like(img[0])

        x_del_indices = {i for i in range(len(cx)) if cx[i] >= x or cx[i] < 0}
        y_del_indices = {i for i in range(len(cy)) if cy[i] >= y or cy[i] < 0}
        del_indices = list(x_del_indices.union(y_del_indices))
        cx = np.delete(cx, del_indices)
        cy = np.delete(cy, del_indices)

        click_map[cx, cy] = 1
        bounding_boxes = []
        for i in range(len(cx)):
            x_start = max(0, cx[i] - bb // 2)
            y_start = max(0, cy[i] - bb // 2)
            x_end = min(x_start + bb, x)
            y_end = min(y_start + bb, y)

            if x_end - x_start != bb:
                x_start = x_end - bb
            if y_end - y_start != bb:
                y_start = y_end - bb
            if x_end - x_start == bb and y_end - y_start == bb:
                bounding_boxes.append([x_start, y_start, x_end, y_end])
            else:
                print(f"Ignore smaller sized bbox ({x_start}, {y_start}, {x_end}, {y_end}) (Min size: {bb}x{bb})")
        return click_map, bounding_boxes

    def get_patches_and_signals(self, img, click_map, bounding_boxes, cx, cy, x, y):
        patches = []

        x_del_indices = {i for i in range(len(cx)) if cx[i] >= x or cx[i] < 0}
        y_del_indices = {i for i in range(len(cy)) if cy[i] >= y or cy[i] < 0}
        del_indices = list(x_del_indices.union(y_del_indices))
        cx = np.delete(cx, del_indices)
        cy = np.delete(cy, del_indices)

        for i, bounding_box in enumerate(bounding_boxes):
            x_start = bounding_box[0]
            y_start = bounding_box[1]
            x_end = bounding_box[2]
            y_end = bounding_box[3]

            patch = img[:, x_start:x_end, y_start:y_end]

            this_click_map = torch.zeros_like(img[0])
            this_click_map[cx[i], cy[i]] = 1

            nuc_points = this_click_map[x_start:x_end, y_start:y_end]
            nuc_points = self._apply_gaussian(nuc_points)

            if self.add_exclusion_map:
                others_click_map = ((click_map - this_click_map) > 0).type(img.dtype)
                other_points = others_click_map[x_start:x_end, y_start:y_end]
                other_points = self._apply_gaussian(other_points)
                patches.append(torch.cat([patch, nuc_points[None], other_points[None]]))
            else:
                patches.append(torch.cat([patch, nuc_points[None]]))

        return torch.stack(patches)

    def _apply_gaussian(self, t):
        if not self.gaussian or torch.count_nonzero(t) == 0:
            return t
        x = GaussianFilter(spatial_dims=2, truncated=self.truncated, sigma=self.sigma)(t.unsqueeze(0).unsqueeze(0))
        return x.squeeze(0).squeeze(0)


class PostFilterLabeld(MapTransform):
    """
    Performs Filtering of Labels on the predicted probability map

    Args:
        thresh: probability threshold for classifying a pixel as a mask
        min_size: min_size objects that will be removed from the image, refer skimage remove_small_objects
        min_hole: min_hole that will be removed from the image, refer skimage remove_small_holes
        do_reconstruction: Boolean Flag, Perform a morphological reconstruction of an image, refer skimage
        allow_missing_keys: don't raise exception if key is missing.
        pred_classes: List of Predicted class for each instance
    """

    def __init__(
        self,
        keys: KeysCollection,
        nuc_points: str = NuclickKeys.NUC_POINTS,
        bounding_boxes: str = NuclickKeys.BOUNDING_BOXES,
        img_height: str = NuclickKeys.IMG_HEIGHT,
        img_width: str = NuclickKeys.IMG_WIDTH,
        thresh: float = 0.33,
        min_size: int = 10,
        min_hole: int = 30,
        do_reconstruction: bool = False,
        allow_missing_keys: bool = False,
        pred_classes: str = NuclickKeys.PRED_CLASSES,
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
        self.pred_classes = pred_classes

    def __call__(self, data):
        d = dict(data)

        pred_classes = d.get(self.pred_classes)
        bounding_boxes = d[self.bounding_boxes]
        x = d[self.img_width]
        y = d[self.img_height]

        for key in self.keys:
            label = d[key].astype(np.uint8)
            masks = self.post_processing(label, self.thresh, self.min_size, self.min_hole)
            d[key] = self.gen_instance_map(masks, bounding_boxes, x, y, pred_classes=pred_classes).astype(np.uint8)
        return d

    def post_processing(self, preds, thresh=0.33, min_size=10, min_hole=30):
        masks = preds > thresh
        for i in range(preds.shape[0]):
            masks[i] = morphology.remove_small_objects(masks[i], min_size=min_size)
            masks[i] = morphology.remove_small_holes(masks[i], area_threshold=min_hole)
        return masks

    def gen_instance_map(self, masks, bounding_boxes, x, y, flatten=True, pred_classes=None):
        instance_map = np.zeros((x, y), dtype=np.uint16)
        for i, mask in enumerate(masks):
            bb = bounding_boxes[i]
            c = pred_classes[i] if pred_classes and i < len(pred_classes) else 1
            c = c if flatten else i + 1

            this_map = instance_map[bb[0] : bb[2], bb[1] : bb[3]]
            this_map = np.where(mask > 0, c, this_map)
            instance_map[bb[0] : bb[2], bb[1] : bb[3]] = this_map

        return instance_map


class AddLabelAsGuidanced(MapTransform):
    """
    Add Label as new guidance channel

    Args:
        source: label/source key which gets added as additional guidance channel
    """

    def __init__(self, keys: KeysCollection, source: str = "label") -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.source = source

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key] if isinstance(d[key], torch.Tensor) else torch.from_numpy(d[key])
            label = d[self.source] if isinstance(d[self.source], torch.Tensor) else torch.from_numpy(d[self.source])

            label = label > 0
            if len(label.shape) < len(image.shape):
                label = label[None]
            image = torch.cat([image, label.type(image.dtype)], dim=len(label.shape) - 3)
            d[key] = image if isinstance(d[key], torch.Tensor) else convert_to_numpy(image)
        return d


class SetLabelClassd(MapTransform):
    """
    Assign class value from the labelmap.  This converts multi-dimension tensor to single scalar tensor.

    Args:
        offset: offset value to be added to the mask value to determine the final class
    """

    def __init__(self, keys: KeysCollection, offset: int = -1) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.offset = offset

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key] if isinstance(d[key], torch.Tensor) else torch.from_numpy(d[key])
            mask_value = int(torch.max(label))
            d[key] = mask_value + self.offset
        return d
