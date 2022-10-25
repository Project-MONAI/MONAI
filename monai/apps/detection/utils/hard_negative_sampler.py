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
from typing import List, Tuple

import torch
from torch import Tensor


class HardNegativeSamplerBase:
    """
    Base class of hard negative sampler.

    Hard negative sampler is used to suppress false positive rate in classification tasks.
    During training, it select negative samples with high prediction scores.

    The training workflow is described as the follows:
    1) forward network and get prediction scores (classification prob/logits) for all the samples;
    2) use hard negative sampler to choose negative samples with high prediction scores and some positive samples;
    3) compute classification loss for the selected samples;
    4) do back propagation.

    Args:
        pool_size: when we need ``num_neg`` hard negative samples, they will be randomly selected from
            ``num_neg * pool_size`` negative samples with the highest prediction scores.
            Larger ``pool_size`` gives more randomness, yet selects negative samples that are less 'hard',
            i.e., negative samples with lower prediction scores.
    """

    def __init__(self, pool_size: float = 10) -> None:
        self.pool_size = pool_size

    def select_negatives(self, negative: Tensor, num_neg: int, fg_probs: Tensor) -> Tensor:
        """
        Select hard negative samples.

        Args:
            negative: indices of all the negative samples, sized (P,),
                where P is the number of negative samples
            num_neg: number of negative samples to sample
            fg_probs: maximum foreground prediction scores (probability) across all the classes
                for each sample, sized (A,), where A is the number of samples.

        Returns:
            binary mask of negative samples to choose, sized (A,),
                where A is the number of samples in one image
        """
        if negative.numel() > fg_probs.numel():
            raise ValueError("The number of negative samples should not be larger than the number of all samples.")

        # sample pool size is ``num_neg * self.pool_size``
        pool = int(num_neg * self.pool_size)
        pool = min(negative.numel(), pool)  # protect against not enough negatives

        # create a sample pool of highest scoring negative samples
        _, negative_idx_pool = fg_probs[negative].to(torch.float32).topk(pool, dim=0, sorted=True)
        hard_negative = negative[negative_idx_pool]

        # select negatives from pool
        perm2 = torch.randperm(hard_negative.numel(), device=hard_negative.device)[:num_neg]
        selected_neg_idx = hard_negative[perm2]

        # output a binary mask with same size of fg_probs that indicates selected negative samples.
        neg_mask = torch.zeros_like(fg_probs, dtype=torch.uint8)
        neg_mask[selected_neg_idx] = 1
        return neg_mask


class HardNegativeSampler(HardNegativeSamplerBase):
    """
    HardNegativeSampler is used to suppress false positive rate in classification tasks.
    During training, it selects negative samples with high prediction scores.

    The training workflow is described as the follows:
    1) forward network and get prediction scores (classification prob/logits) for all the samples;
    2) use hard negative sampler to choose negative samples with high prediction scores and some positive samples;
    3) compute classification loss for the selected samples;
    4) do back propagation.

    Args:
        batch_size_per_image: number of training samples to be randomly selected per image
        positive_fraction: percentage of positive elements in the selected samples
        min_neg: minimum number of negative samples to select if possible.
        pool_size: when we need ``num_neg`` hard negative samples, they will be randomly selected from
            ``num_neg * pool_size`` negative samples with the highest prediction scores.
            Larger ``pool_size`` gives more randomness, yet selects negative samples that are less 'hard',
            i.e., negative samples with lower prediction scores.
    """

    def __init__(
        self, batch_size_per_image: int, positive_fraction: float, min_neg: int = 1, pool_size: float = 10
    ) -> None:
        super().__init__(pool_size=pool_size)
        self.min_neg = min_neg
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        logging.info("Sampling hard negatives on a per batch basis")

    def __call__(self, target_labels: List[Tensor], concat_fg_probs: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Select positives and hard negatives from list samples per image.
        Hard negative sampler will be applied to each image independently.

        Args:
            target_labels: list of labels per image.
                For image i in the batch, target_labels[i] is a Tensor sized (A_i,),
                where A_i is the number of samples in image i.
                Positive samples have positive labels, negative samples have label 0.
            concat_fg_probs: concatenated maximum foreground probability for all the images, sized (R,),
                where R is the sum of all samples inside one batch, i.e., R = A_0 + A_1 + ...

        Returns:
            - list of binary mask for positive samples
            - list of binary mask for negative samples

        Example:
            .. code-block:: python

                sampler = HardNegativeSampler(
                    batch_size_per_image=6, positive_fraction=0.5, min_neg=1, pool_size=2
                )
                # two images with different number of samples
                target_labels = [ torch.tensor([0,1]), torch.tensor([1,0,2,1])]
                concat_fg_probs = torch.rand(6)
                pos_idx_list, neg_idx_list = sampler(target_labels, concat_fg_probs)
        """
        samples_per_image = [samples_in_image.shape[0] for samples_in_image in target_labels]
        fg_probs = concat_fg_probs.split(samples_per_image, 0)
        return self.select_samples_img_list(target_labels, fg_probs)

    def select_samples_img_list(
        self, target_labels: List[Tensor], fg_probs: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Select positives and hard negatives from list samples per image.
        Hard negative sampler will be applied to each image independently.

        Args:
            target_labels: list of labels per image.
                For image i in the batch, target_labels[i] is a Tensor sized (A_i,),
                where A_i is the number of samples in image i.
                Positive samples have positive labels, negative samples have label 0.
            fg_probs: list of maximum foreground probability per images,
                For image i in the batch, target_labels[i] is a Tensor sized (A_i,),
                where A_i is the number of samples in image i.

        Returns:
            - list of binary mask for positive samples
            - list binary mask for negative samples

        Example:
            .. code-block:: python

                sampler = HardNegativeSampler(
                    batch_size_per_image=6, positive_fraction=0.5, min_neg=1, pool_size=2
                )
                # two images with different number of samples
                target_labels = [ torch.tensor([0,1]), torch.tensor([1,0,2,1])]
                fg_probs = [ torch.rand(2), torch.rand(4)]
                pos_idx_list, neg_idx_list = sampler.select_samples_img_list(target_labels, fg_probs)
        """
        pos_idx = []
        neg_idx = []

        if len(target_labels) != len(fg_probs):
            raise ValueError(
                "Require len(target_labels) == len(fg_probs). "
                f"Got len(target_labels)={len(target_labels)}, len(fg_probs)={len(fg_probs)}."
            )
        for labels_per_img, fg_probs_per_img in zip(target_labels, fg_probs):
            pos_idx_per_image_mask, neg_idx_per_image_mask = self.select_samples_per_img(
                labels_per_img, fg_probs_per_img
            )
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx

    def select_samples_per_img(self, labels_per_img: Tensor, fg_probs_per_img: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Select positives and hard negatives from samples.

        Args:
            labels_per_img: labels, sized (A,).
                Positive samples have positive labels, negative samples have label 0.
            fg_probs_per_img: maximum foreground probability, sized (A,)

        Returns:
            - binary mask for positive samples, sized (A,)
            - binary mask for negative samples, sized (A,)

        Example:
            .. code-block:: python

                sampler = HardNegativeSampler(
                    batch_size_per_image=6, positive_fraction=0.5, min_neg=1, pool_size=2
                )
                # two images with different number of samples
                target_labels = torch.tensor([1,0,2,1])
                fg_probs = torch.rand(4)
                pos_idx, neg_idx = sampler.select_samples_per_img(target_labels, fg_probs)
        """
        # for each image, find positive sample indices and negative sample indices
        if labels_per_img.numel() != fg_probs_per_img.numel():
            raise ValueError("labels_per_img and fg_probs_per_img should have same number of elements.")

        positive = torch.where(labels_per_img >= 1)[0]
        negative = torch.where(labels_per_img == 0)[0]

        num_pos = self.get_num_pos(positive)
        pos_idx_per_image_mask = self.select_positives(positive, num_pos, labels_per_img)

        num_neg = self.get_num_neg(negative, num_pos)
        neg_idx_per_image_mask = self.select_negatives(negative, num_neg, fg_probs_per_img)

        return pos_idx_per_image_mask, neg_idx_per_image_mask

    def get_num_pos(self, positive: torch.Tensor) -> int:
        """
        Number of positive samples to draw

        Args:
            positive: indices of positive samples

        Returns:
            number of positive sample
        """
        # positive sample sampling
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
        # always assume at least one pos sample was sampled
        num_neg = int(max(1, num_pos) * abs(1 - 1.0 / float(self.positive_fraction)))
        # protect against not enough negative examples and sample at least self.min_neg if possible
        num_neg = min(negative.numel(), max(num_neg, self.min_neg))
        return num_neg

    def select_positives(self, positive: Tensor, num_pos: int, labels: Tensor) -> Tensor:
        """
        Select positive samples

        Args:
            positive: indices of positive samples, sized (P,),
                where P is the number of positive samples
            num_pos: number of positive samples to sample
            labels: labels for all samples, sized (A,),
                where A is the number of samples.

        Returns:
            binary mask of positive samples to choose, sized (A,),
                where A is the number of samples in one image
        """
        if positive.numel() > labels.numel():
            raise ValueError("The number of positive samples should not be larger than the number of all samples.")

        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        pos_idx_per_image = positive[perm1]

        # output a binary mask with same size of labels that indicates selected positive samples.
        pos_idx_per_image_mask = torch.zeros_like(labels, dtype=torch.uint8)
        pos_idx_per_image_mask[pos_idx_per_image] = 1
        return pos_idx_per_image_mask
