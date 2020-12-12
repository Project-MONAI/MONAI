# Copyright 2020 MONAI Consortium
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
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import numpy as np

from monai.transforms.compose import Randomizable, Transform
from monai.utils import min_version, optional_import

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)
distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")
gaussian_filter, _ = optional_import("scipy.ndimage", name="gaussian_filter")


class AddInitialSeedPointd(Randomizable, Transform):
    def __init__(self, label='label', guidance='guidance', dim=2, connected_regions=6):
        self.label = label
        self.guidance = guidance
        self.dim = dim
        self.connected_regions = connected_regions

    def randomize(self, data=None):
        pass

    def _apply(self, label):
        label = (label > 0.5).astype(np.float32)

        blobs_labels = measure.label(label.astype(int), background=0) if self.dim == 2 else label
        assert np.max(blobs_labels) > 0, "Not a valid Label"

        default_guidance = [-1] * (self.dim + 1)
        pos_guidance = []
        for ridx in range(1, 2 if self.dim == 3 else self.connected_regions):
            if self.dim == 2:
                label = (blobs_labels == ridx).astype(np.float32)
                if np.sum(label) == 0:
                    pos_guidance.append(default_guidance)
                    continue

            distance = distance_transform_cdt(label).flatten()
            probability = np.exp(distance) - 1.0

            idx = np.where(label.flatten() > 0)[0]
            seed = np.random.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
            g[0] = dst[0]
            pos_guidance.append(g)

        return np.asarray([pos_guidance, [default_guidance] * len(pos_guidance)])

    def __call__(self, data):
        data[self.guidance] = self._apply(data[self.label])
        return data


class AddGuidanceSignald(Transform):
    def __init__(self, image='image', guidance='guidance', sigma=2, dim=2, number_intensity_ch=1, batched=False):
        self.image = image
        self.guidance = guidance
        self.sigma = sigma
        self.dim = dim
        self.number_intensity_ch = number_intensity_ch
        self.batched = batched

    def _get_signal(self, image, guidance):
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        if self.dim == 3:
            signal = np.zeros((len(guidance), image.shape[-3], image.shape[-2], image.shape[-1]), dtype=np.float32)
        else:
            signal = np.zeros((len(guidance), image.shape[-2], image.shape[-1]), dtype=np.float32)

        for i in range(len(guidance)):
            for point in guidance[i]:
                if np.any(np.asarray(point) < 0):
                    continue

                # print('{}:: Point: {}'.format('-VE' if i else '+VE', np.asarray(point)))
                if self.dim == 3:
                    signal[i, int(point[-3]), int(point[-2]), int(point[-1])] = 1.0
                else:
                    signal[i, int(point[-2]), int(point[-1])] = 1.0

            if np.max(signal[i]) > 0:
                signal[i] = gaussian_filter(signal[i], sigma=self.sigma)
                signal[i] = (signal[i] - np.min(signal[i])) / (np.max(signal[i]) - np.min(signal[i]))
        return signal

    def _apply(self, image, guidance):
        if not self.batched:
            signal = self._get_signal(image, guidance)
            return np.concatenate([image, signal], axis=0)

        images = []
        for i, g in zip(image, guidance):
            i = i[0:0 + self.number_intensity_ch, ...]
            signal = self._get_signal(i, g)
            images.append(np.concatenate([i, signal], axis=0))
        return images

    def __call__(self, data):
        image = data[self.image]
        guidance = data[self.guidance]

        data[self.image] = self._apply(image, guidance)
        return data


class FindDiscrepancyRegionsd(Transform):
    def __init__(self, label='label', pred='pred', discrepancy='discrepancy', batched=True):
        self.label = label
        self.pred = pred
        self.discrepancy = discrepancy
        self.batched = batched

    @staticmethod
    def disparity(label, pred):
        label = (label > 0.5).astype(np.float32)
        pred = (pred > 0.5).astype(np.float32)
        disparity = label - pred

        pos_disparity = (disparity > 0).astype(np.float32)
        neg_disparity = (disparity < 0).astype(np.float32)
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        if not self.batched:
            return self.disparity(label, pred)

        disparity = []
        for la, pr in zip(label, pred):
            disparity.append(self.disparity(la, pr))
        return disparity

    def __call__(self, data):
        label = data[self.label]
        pred = data[self.pred]

        data[self.discrepancy] = self._apply(label, pred)
        return data


class AddRandomGuidanced(Randomizable, Transform):
    def __init__(self, guidance='guidance', discrepancy='discrepancy', probability='probability', dim=2, batched=True):
        self.guidance = guidance
        self.discrepancy = discrepancy
        self.probability = probability
        self.dim = dim
        self.batched = batched

    def randomize(self, data=None):
        pass

    @staticmethod
    def find_guidance(discrepancy):
        distance = distance_transform_cdt(discrepancy).flatten()
        probability = np.exp(distance) - 1.0
        idx = np.where(discrepancy.flatten() > 0)[0]

        if np.sum(discrepancy > 0) > 0:
            seed = np.random.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, discrepancy.shape)).transpose().tolist()[0]
            g[0] = dst[0]
            return g
        return None

    @staticmethod
    def add_guidance(discrepancy, probability):
        will_interact = np.random.choice([True, False], p=[probability, 1.0 - probability])
        if not will_interact:
            return None, None

        pos_discr = discrepancy[0]
        neg_discr = discrepancy[1]

        can_be_positive = np.sum(pos_discr) > 0
        can_be_negative = np.sum(neg_discr) > 0
        correct_pos = np.sum(pos_discr) >= np.sum(neg_discr)

        if correct_pos and can_be_positive:
            return AddRandomGuidanced.find_guidance(pos_discr), None

        if not correct_pos and can_be_negative:
            return None, AddRandomGuidanced.find_guidance(neg_discr)
        return None, None

    def _apply(self, guidance, discrepancy, probability):
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        default_guidance = [-1] * (self.dim + 1)

        if not self.batched:
            pos, neg = self.add_guidance(discrepancy, probability)
            if pos:
                guidance[0].append(pos)
                guidance[1].append(default_guidance)
            if neg:
                guidance[0].append(default_guidance)
                guidance[1].append(neg)
        else:
            for g, d, p in zip(guidance, discrepancy, probability):
                pos, neg = self.add_guidance(d, p)
                if pos:
                    g[0].append(pos)
                    g[1].append(default_guidance)
                if neg:
                    g[0].append(default_guidance)
                    g[1].append(neg)
        return np.asarray(guidance)

    def __call__(self, data):
        guidance = data[self.guidance]
        discrepancy = data[self.discrepancy]
        probability = data[self.probability]

        data[self.guidance] = self._apply(guidance, discrepancy, probability)
        return data
