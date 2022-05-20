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

import cv2
import numpy as np
import skimage
from skimage.morphology import remove_small_objects

from monai.config import KeysCollection
from monai.transforms import MapTransform, RandomizableTransform, Transform


class FlattenLabeld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            _, labels, _, _ = cv2.connectedComponentsWithStats(d[key], 4, cv2.CV_32S)
            d[key] = labels.astype(np.uint8)
        return d


class ExtractPatchd(MapTransform):
    def __init__(self, keys: KeysCollection, centroid_key="centroid", patch_size=128):
        super().__init__(keys)
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
            d[key] = self.pad_to_shape(cropped, roi_size)
        return d

    @staticmethod
    def bbox(patch_size, centroid, size):
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

    @staticmethod
    def pad_to_shape(img, shape):
        img_shape = img.shape[-2:]
        s_diff = np.array(shape) - np.array(img_shape)
        diff = [(0, 0), (0, s_diff[0]), (0, s_diff[1])]
        return np.pad(img, diff, mode="constant", constant_values=0)


class SplitLabeld(Transform):
    def __init__(self, label="label", others="others", mask_value="mask_value", min_area=5):
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

    @staticmethod
    def _mask_relabeling(mask, min_area=5):
        res = np.zeros_like(mask)
        for l in np.unique(mask):
            if l == 0:
                continue

            m = skimage.measure.label(mask == l, connectivity=1)
            for stat in skimage.measure.regionprops(m):
                if stat.area > min_area:
                    res[stat.coords[:, 0], stat.coords[:, 1]] = l
        return res


class FilterImaged(MapTransform):
    def __init__(self, keys: KeysCollection, min_size: int = 500):
        super().__init__(keys)
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
        rem_sm = remove_small_objects(img_np.astype(bool), min_size=min_size)
        mask_percentage = self.mask_percent(rem_sm)
        if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
            new_min_size = round(min_size / 2)
            rem_sm = self.filter_remove_small_objects(img_np, new_min_size, avoid_overmask, overmask_thresh)
        return rem_sm


class AddPointGuidanceSignald(RandomizableTransform):
    def __init__(self, image="image", label="label", others="others", drop_rate=0.5, jitter_range=3):
        super().__init__()

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

    @staticmethod
    def inclusion_map(mask):
        point_mask = np.zeros_like(mask)
        indices = np.argwhere(mask > 0)
        if len(indices) > 0:
            idx = np.random.randint(0, len(indices))
            point_mask[indices[idx, 0], indices[idx, 1]] = 1

        return point_mask

    @staticmethod
    def exclusion_map(others, jitter_range=3, drop_rate=0.5):
        point_mask = np.zeros_like(others)
        if drop_rate == 1.0:
            return point_mask

        max_x = point_mask.shape[0] - 1
        max_y = point_mask.shape[1] - 1
        stats = skimage.measure.regionprops(others)
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
