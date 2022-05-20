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
This script is modified from  https://github.com/MIC-DKFZ/nnDetection/blob/main/nndet/core/boxes/sampler.py

Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from abc import ABC
from typing import List

import torch
from loguru import logger
from torch import Tensor
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler


class AbstractSampler(ABC):
    def __call__(self, target_labels: List[Tensor], fg_probs: Tensor):
        """
        Select positive and negative anchors

        Args:
            target_labels: labels for each anchor per image, List[[A]],
                where A is the number of anchors in one image
            fg_probs: maximum foreground probability per anchor, [R]
                where R is the sum of all anchors inside one batch

        Returns:
            List[Tensor]: binary mask for positive anchors, List[[A]]
            List[Tensor]: binary mask for negative anchors, List[[A]]
        """
        raise NotImplementedError


class NegativeSampler(BalancedPositiveNegativeSampler, AbstractSampler):
    def __call__(self, target_labels: List[Tensor], fg_probs: Tensor):
        """
        Randomly sample negatives and positives until batch_size_per_img
        is reached
        If not enough positive samples are found, it will be padded with
        negative samples
        """
        return super().__call__(target_labels, fg_probs)


class HardNegativeSamplerMixin(ABC):
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
        Select negative anchors

        Args:
            negative: indices of negative anchors [P],
                where P is the number of negative anchors
            num_neg: number of negative anchors to sample
            img_labels: labels for all anchors in a image [A],
                where A is the number of anchors in one image
            img_fg_probs: maximum foreground probability per anchor [A],
                where A is the the number of anchors in one image

        Returns:
            Tensor: binary mask of negative anchors to choose [A],
                where A is the the number of anchors in one image
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

    def __call__(self, target_labels: List[Tensor], fg_probs: Tensor):
        """
        Select hard negatives from list anchors per image

        Args:
            target_labels: labels for each anchor per image, List[[A]],
                where A is the number of anchors in one image
            fg_probs: maximum foreground probability per anchor, [R]
                where R is the sum of all anchors inside one batch

        Returns:
            List[Tensor]: binary mask for positive anchors, List[[A]]
            List[Tensor]: binary mask for negative anchors, List[[A]]
        """
        anchors_per_image = [anchors_in_image.shape[0] for anchors_in_image in target_labels]
        fg_probs = fg_probs.split(anchors_per_image, 0)

        pos_idx = []
        neg_idx = []
        for img_labels, img_fg_probs in zip(target_labels, fg_probs):
            positive = torch.where(img_labels >= 1)[0]
            negative = torch.where(img_labels == 0)[0]

            num_pos = self.get_num_pos(positive)
            pos_idx_per_image_mask = self.select_positives(positive, num_pos, img_labels, img_fg_probs)
            pos_idx.append(pos_idx_per_image_mask)

            num_neg = self.get_num_neg(negative, num_pos)
            neg_idx_per_image_mask = self.select_negatives(negative, num_neg, img_labels, img_fg_probs)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx

    def get_num_pos(self, positive: torch.Tensor) -> int:
        """
        Number of positive samples to draw

        Args:
            positive: indices of positive anchors

        Returns:
            int: number of postive sample
        """
        # positive anchor sampling
        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        return num_pos

    def get_num_neg(self, negative: torch.Tensor, num_pos: int) -> int:
        """
        Sample enough negatives to fill up :param:`self.batch_size_per_image`

        Args:
            negative: indices of positive anchors
            num_pos: number of positive samples to draw

        Returns:
            int: number of negative samples
        """
        # always assume at least one pos anchor was sampled
        num_neg = int(max(1, num_pos) * abs(1 - 1.0 / float(self.positive_fraction)))
        # protect against not enough negative examples and sample at least one neg if possible
        num_neg = min(negative.numel(), max(num_neg, self.min_neg))
        return num_neg

    def select_positives(self, positive: Tensor, num_pos: int, img_labels: Tensor, img_fg_probs: Tensor):
        """
        Select positive anchors

        Args:
            positive: indices of positive anchors [P],
                where P is the number of positive anchors
            num_pos: number of positive anchors to sample
            img_labels: labels for all anchors in a image [A],
                where A is the number of anchors in one image
            img_fg_probs: maximum foreground probability per anchor [A],
                where A is the the number of anchors in one image

        Returns:
            Tensor: binary mask of positive anchors to choose [A],
                where A is the the number of anchors in one image
        """
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        pos_idx_per_image = positive[perm1]
        pos_idx_per_image_mask = torch.zeros_like(img_labels, dtype=torch.uint8)
        pos_idx_per_image_mask[pos_idx_per_image] = 1
        return pos_idx_per_image_mask


class HardNegativeSamplerBatched(HardNegativeSampler):
    """
    Samples negatives and positives on a per batch basis
    (default sampler only does this on a per image basis)

    Note:
        :attr:`batch_size_per_image` is manipulated to sample the correct
        number of samples per batch, use :attr:`_batch_size_per_image`
        to get the number of anchors per image
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float, min_neg: int = 0, pool_size: float = 10):
        """
        Args:
            batch_size_per_image: number of elements to be selected per image
            positive_fraction: percentage of positive elements per batch
            pool_size: hard negatives are sampled from a pool of size:
                batch_size_per_image * (1 - positive_fraction) * pool_size
        """
        super().__init__(
            min_neg=min_neg,
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            pool_size=pool_size,
        )
        self._batch_size_per_image = batch_size_per_image
        logger.info("Sampling hard negatives on a per batch basis")

    def __call__(self, target_labels: List[Tensor], fg_probs: Tensor):
        """
        Select hard negatives from list anchors per image

        Args:
            target_labels: labels for each anchor per image, List[[A]],
                where A is the number of anchors in one image
            fg_probs: maximum foreground probability per anchor, [R]
                where R is the sum of all anchors inside one batch

        Returns:
            List[Tensor]: binary mask for positive anchors, List[[A]]
            List[Tensor]: binary mask for negative anchors, List[[A]]
        """
        batch_size = len(target_labels)
        self.batch_size_per_image = self._batch_size_per_image * batch_size

        target_labels_batch = torch.cat(target_labels, dim=0)

        positive = torch.where(target_labels_batch >= 1)[0]
        negative = torch.where(target_labels_batch == 0)[0]

        num_pos = self.get_num_pos(positive)
        pos_idx = self.select_positives(positive, num_pos, target_labels_batch, fg_probs)

        num_neg = self.get_num_neg(negative, num_pos)
        neg_idx = self.select_negatives(negative, num_neg, target_labels_batch, fg_probs)

        # Comb Head with sampling concatenates masks after sampling so do not split them here
        # anchors_per_image = [anchors_in_image.shape[0] for anchors_in_image in target_labels]
        # return pos_idx.split(anchors_per_image, 0), neg_idx.split(anchors_per_image, 0)
        return [pos_idx], [neg_idx]


class BalancedHardNegativeSampler(HardNegativeSampler):
    def get_num_neg(self, negative: torch.Tensor, num_pos: int) -> int:
        """
        Sample same number of negatives as positives but at least one

        Args:
            negative: indices of positive anchors
            num_pos: number of positive samples to draw

        Returns:
            int: number of negative samples
        """
        # protect against not enough negative examples and sample at least one neg if possible
        num_neg = min(negative.numel(), max(num_pos, 1))
        return num_neg


class HardNegativeSamplerFgAll(HardNegativeSamplerMixin):
    def __init__(self, negative_ratio: float = 1, pool_size: float = 10):
        """
        Use all positive anchors for loss and sample corresponding number
        of hard negatives

        Args:
            negative_ratio: ratio of negative to positive sample;
                (samples negative_ratio * positive_anchors examples)
            pool_size: hard negatives are sampled from a pool of size:
                batch_size_per_image * (1 - positive_fraction) * pool_size
        """
        super().__init__(pool_size=pool_size)
        self.negative_ratio = negative_ratio

    def __call__(self, target_labels: List[Tensor], fg_probs: Tensor):
        """
        Select hard negatives from list anchors per image

        Args:
            target_labels: labels for each anchor per image, List[[A]],
                where A is the number of anchors in one image
            fg_probs: maximum foreground probability per anchor, [R]
                where R is the sum of all anchors inside one batch

        Returns:
            List[Tensor]: binary mask for positive anchors, List[[A]]
            List[Tensor]: binary mask for negative anchors, List[[A]]
        """
        anchors_per_image = [anchors_in_image.shape[0] for anchors_in_image in target_labels]
        fg_probs = fg_probs.split(anchors_per_image, 0)

        pos_idx = []
        neg_idx = []
        for img_labels, img_fg_probs in zip(target_labels, fg_probs):
            negative = torch.where(img_labels == 0)[0]

            # positive anchor sampling
            pos_idx_per_image_mask = (img_labels >= 1).to(dtype=torch.uint8)
            pos_idx.append(pos_idx_per_image_mask)

            num_neg = int(self.negative_ratio * pos_idx_per_image_mask.sum())
            # protect against not enough negative examples and sample at least one neg if possible
            num_neg = min(negative.numel(), max(num_neg, 1))
            neg_idx_per_image_mask = self.select_negatives(negative, num_neg, img_labels, img_fg_probs)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
