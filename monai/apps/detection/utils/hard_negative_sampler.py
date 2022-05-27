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

# =========================================================================
# Adapted from https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/sampler.py
# which has the following license...
# https://github.com/MIC-DKFZ/nnDetection/blob/main/LICENSE
#
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The functions in this script are adapted from nnDetection,
https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/sampler.py
"""

import logging
from abc import ABC
from typing import List

import torch
from torch import Tensor


class HardNegativeSamplerMixin(ABC):
    """
    Base class of hard negative sampler.
    """
    def __init__(self, pool_size: float = 10):
        """
        Create a pool from the highest scoring false positives and sample
        defined number of negatives from it

        Args:
            pool_size: hard negatives are sampled from a pool of size:
                batch_size_per_image * (1 - positive_fraction) * pool_size
        """
        self.pool_size = pool_size

    def select_negatives(self, negative: Tensor, num_neg: int, img_labels: Tensor, img_fg_probs: Tensor):
        """
        Select negative samples

        Args:
            negative: indices of negative samples [P],
                where P is the number of negative samples
            num_neg: number of negative samples to sample
            img_labels: labels for all samples in a image [A],
                where A is the number of samples in one image
            img_fg_probs: maximum foreground probability per anchor [A],
                where A is the the number of samples in one image

        Returns:
            binary mask of negative samples to choose [A],
                where A is the the number of samples in one image
        """
        pool = int(num_neg * self.pool_size)
        pool = min(negative.numel(), pool)  # protect against not enough negatives

        # select pool of highest scoring false positives
        _, negative_idx_pool = img_fg_probs[negative].topk(pool, dim=0, sorted=True)
        negative = negative[negative_idx_pool]

        # select negatives from pool
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
        neg_idx_per_image = negative[perm2]

        neg_idx_per_image_mask = torch.zeros_like(img_labels, dtype=torch.uint8)
        neg_idx_per_image_mask[neg_idx_per_image] = 1
        return neg_idx_per_image_mask


class HardNegativeSampler(HardNegativeSamplerMixin):
    """
    Hard negative sampler.
    """
    def __init__(self, batch_size_per_image: int, positive_fraction: float, min_neg: int = 0, pool_size: float = 10):
        """
        Created a pool from the highest scoring false positives and sample
        defined number of negatives from it

        Args:
            batch_size_per_image: number of elements to be selected per image
            positive_fraction: percentage of positive elements per batch
            pool_size: hard negatives are sampled from a pool of size:
                batch_size_per_image * (1 - positive_fraction) * pool_size
        """
        super().__init__(pool_size=pool_size)
        self.min_neg = min_neg
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        logging.info("Sampling hard negatives on a per batch basis")

    def __call__(self, target_labels: List[Tensor], fg_probs: Tensor):
        """
        Select hard negatives from list samples per image

        Args:
            target_labels: labels for each anchor per image, List[[A]],
                where A is the number of samples in one image
            fg_probs: maximum foreground probability per anchor, [R]
                where R is the sum of all samples inside one batch

        Returns:
            - binary mask for positive samples, List[[A]]
            - binary mask for negative samples, List[[A]]
        """
        samples_per_image = [samples_in_image.shape[0] for samples_in_image in target_labels]
        fg_probs = fg_probs.split(samples_per_image, 0)

        pos_idx = []
        neg_idx = []
        for img_labels, img_fg_probs in zip(target_labels, fg_probs):
            positive = torch.where(img_labels >= 1)[0]
            negative = torch.where(img_labels == 0)[0]

            num_pos = self.get_num_pos(positive)
            pos_idx_per_image_mask = self.select_positives(positive, num_pos, img_labels)
            pos_idx.append(pos_idx_per_image_mask)

            num_neg = self.get_num_neg(negative, num_pos)
            neg_idx_per_image_mask = self.select_negatives(negative, num_neg, img_labels, img_fg_probs)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx

    def get_num_pos(self, positive: torch.Tensor) -> int:
        """
        Number of positive samples to draw

        Args:
            positive: indices of positive samples

        Returns:
            number of postive sample
        """
        # positive anchor sampling
        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        return num_pos

    def get_num_neg(self, negative: torch.Tensor, num_pos: int) -> int:
        """
        Sample enough negatives to fill up ``self.batch_size_per_image``

        Args:
            negative: indices of positive samples
            num_pos: number of positive samples to draw

        Returns:
            number of negative samples
        """
        # always assume at least one pos anchor was sampled
        num_neg = int(max(1, num_pos) * abs(1 - 1.0 / float(self.positive_fraction)))
        # protect against not enough negative examples and sample at least one neg if possible
        num_neg = min(negative.numel(), max(num_neg, self.min_neg))
        return num_neg

    def select_positives(self, positive: Tensor, num_pos: int, img_labels: Tensor):
        """
        Select positive samples

        Args:
            positive: indices of positive samples [P],
                where P is the number of positive samples
            num_pos: number of positive samples to sample
            img_labels: labels for all samples in a image [A],
                where A is the number of samples in one image

        Returns:
            binary mask of positive samples to choose [A],
                where A is the the number of samples in one image
        """
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        pos_idx_per_image = positive[perm1]
        pos_idx_per_image_mask = torch.zeros_like(img_labels, dtype=torch.uint8)
        pos_idx_per_image_mask[pos_idx_per_image] = 1
        return pos_idx_per_image_mask
