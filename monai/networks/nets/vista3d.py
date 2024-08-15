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
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import monai
from monai.networks.blocks import MLPBlock, UnetrBasicBlock
from monai.networks.nets import SegResNetDS2
from monai.transforms.utils import convert_points_to_disc
from monai.transforms.utils import get_largest_connected_component_mask_point as lcc
from monai.transforms.utils import sample_points_from_label
from monai.utils import optional_import, unsqueeze_left, unsqueeze_right

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = ["VISTA3D", "vista3d132"]


def vista3d132(encoder_embed_dim: int = 48, in_channels: int = 1):
    """
    Exact VISTA3D network configuration used in https://arxiv.org/abs/2406.05285>`_.
    The model treats class index larger than 132 as zero-shot.

    Args:
        encoder_embed_dim: hidden dimension for encoder.
        in_channels: input channel number.
    """
    segresnet = SegResNetDS2(
        in_channels=in_channels,
        blocks_down=(1, 2, 2, 4, 4),
        norm="instance",
        out_channels=encoder_embed_dim,
        init_filters=encoder_embed_dim,
        dsdepth=1,
    )
    point_head = PointMappingSAM(feature_size=encoder_embed_dim, n_classes=512, last_supported=132)
    class_head = ClassMappingClassify(n_classes=512, feature_size=encoder_embed_dim, use_mlp=True)
    vista = VISTA3D(image_encoder=segresnet, class_head=class_head, point_head=point_head)
    return vista


class VISTA3D(nn.Module):
    """
    VISTA3D based on:
        `VISTA3D: Versatile Imaging SegmenTation and Annotation model for 3D Computed Tomography
        <https://arxiv.org/abs/2406.05285>`_.

    Args:
        image_encoder: image encoder backbone for feature extraction.
        class_head: class head used for class index based segmentation
        point_head: point head used for interactive segmetnation
    """

    def __init__(self, image_encoder: nn.Module, class_head: nn.Module, point_head: nn.Module):
        super().__init__()
        self.image_encoder = image_encoder
        self.class_head = class_head
        self.point_head = point_head
        self.image_embeddings = None
        self.auto_freeze = False
        self.point_freeze = False
        self.NINF_VALUE = -9999
        self.PINF_VALUE = 9999

    def get_foreground_class_count(self, class_vector: torch.Tensor | None, point_coords: torch.Tensor | None) -> int:
        """Get number of foreground classes based on class and point prompt."""
        if class_vector is None:
            if point_coords is None:
                raise ValueError("class_vector and point_coords cannot be both None.")
            return point_coords.shape[0]
        else:
            return class_vector.shape[0]

    def convert_point_label(
        self,
        point_label: torch.Tensor,
        label_set: Sequence[int] | None = None,
        special_index: Sequence[int] = (23, 24, 25, 26, 27, 57, 128),
    ):
        """
        Convert point label based on its class prompt. For special classes defined in special index,
        the positive/negative point label will be converted from 1/0 to 3/2. The purpose is to separate those
        classes with ambiguous classes.

        Args:
            point_label: the point label tensor, [B, N].
            label_set: the label index matching the indexes in labels. If labels are mapped to global index using RelabelID,
                this label_set should be global mapped index. If labels are not mapped to global index, e.g. in zero-shot
                evaluation, this label_set should be the original index.
            special_index: the special class index that needs to be converted.
        """
        if label_set is None:
            return point_label
        if not point_label.shape[0] == len(label_set):
            raise ValueError("point_label and label_set must have the same length.")

        for i in range(len(label_set)):
            if label_set[i] in special_index:
                for j in range(len(point_label[i])):
                    point_label[i, j] = point_label[i, j] + 2 if point_label[i, j] > -1 else point_label[i, j]
        return point_label

    def sample_points_patch_val(
        self,
        labels: torch.Tensor,
        patch_coords: Sequence[slice],
        label_set: Sequence[int],
        use_center: bool = True,
        mapped_label_set: Sequence[int] | None = None,
        max_ppoint: int = 1,
        max_npoint: int = 0,
    ):
        """
        Sample points for patch during sliding window validation. Only used for point only validation.

        Args:
            labels: shape [1, 1, H, W, D].
            patch_coords: a sequence of sliding window slice objects.
            label_set: local index, must match values in labels.
            use_center: sample points from the center.
            mapped_label_set: global index, it is used to identify special classes and is the global index
                for the sampled points.
            max_ppoint/max_npoint: positive points and negative points to sample.
        """
        point_coords, point_labels = sample_points_from_label(
            labels[patch_coords],
            label_set,
            max_ppoint=max_ppoint,
            max_npoint=max_npoint,
            device=labels.device,
            use_center=use_center,
        )
        point_labels = self.convert_point_label(point_labels, mapped_label_set)
        return (point_coords, point_labels, torch.tensor(label_set).to(point_coords.device).unsqueeze(-1))

    def update_point_to_patch(
        self, patch_coords: Sequence[slice], point_coords: torch.Tensor, point_labels: torch.Tensor
    ):
        """
        Update point_coords with respect to patch coords.
        If point is outside of the patch, remove the coordinates and set label to -1.

        Args:
            patch_coords: a sequence of the python slice objects representing the patch coordinates during sliding window inference.
                This value is passed from sliding_window_inferer.
            point_coords: point coordinates, [B, N, 3].
            point_labels: point labels, [B, N].
        """
        patch_ends = [patch_coords[-3].stop, patch_coords[-2].stop, patch_coords[-1].stop]
        patch_starts = [patch_coords[-3].start, patch_coords[-2].start, patch_coords[-1].start]
        # update point coords
        patch_starts_tensor = unsqueeze_left(torch.tensor(patch_starts, device=point_coords.device), 2)
        patch_ends_tensor = unsqueeze_left(torch.tensor(patch_ends, device=point_coords.device), 2)
        # [1 N 1]
        indices = torch.logical_and(
            ((point_coords - patch_starts_tensor) > 0).all(2), ((patch_ends_tensor - point_coords) > 0).all(2)
        )
        # check if it's within patch coords
        point_coords = point_coords.clone() - patch_starts_tensor
        point_labels = point_labels.clone()
        if indices.any():
            point_labels[~indices] = -1
            point_coords[~indices] = 0
            # also remove padded points, mainly used for inference.
            not_pad_indices = (point_labels != -1).any(0)
            point_coords = point_coords[:, not_pad_indices]
            point_labels = point_labels[:, not_pad_indices]
            return point_coords, point_labels
        return None, None

    def connected_components_combine(
        self,
        logits: torch.Tensor,
        point_logits: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mapping_index: torch.Tensor,
        thred: float = 0.5,
    ):
        """
        Combine auto results with point click response. The auto results have shape [B, 1, H, W, D] which means B foreground masks
        from a single image patch.
        Out of those B foreground masks, user may add points to a subset of B1 foreground masks for editing.
        mapping_index represents the correspondence between B and B1.
        For mapping_index with point clicks, NaN values in logits will be replaced with point_logits. Meanwhile, the added/removed
        region in point clicks must be updated by the lcc function.
        Notice, if a positive point is within logits/prev_mask, the components containing the positive point will be added.

        Args:
            logits: automatic branch results, [B, 1, H, W, D].
            point_logits: point branch results, [B1, 1, H, W, D].
            point_coords: point coordinates, [B1, N, 3].
            point_labels: point labels, [B1, N].
            mapping_index: [B].
            thred: the threshold to convert logits to binary.
        """
        logits = logits.as_tensor() if isinstance(logits, monai.data.MetaTensor) else logits
        _logits = logits[mapping_index]
        inside = []
        for i in range(_logits.shape[0]):
            inside.append(
                np.any(
                    [
                        _logits[i, 0, p[0], p[1], p[2]].item() > 0
                        for p in point_coords[i].cpu().numpy().round().astype(int)
                    ]
                )
            )
        inside_tensor = torch.tensor(inside).to(logits.device)
        nan_mask = torch.isnan(_logits)
        # _logits are converted to binary [B1, 1, H, W, D]
        _logits = torch.nan_to_num(_logits, nan=self.NINF_VALUE).sigmoid()
        pos_region = point_logits.sigmoid() > thred
        diff_pos = torch.logical_and(torch.logical_or(_logits <= thred, unsqueeze_right(inside_tensor, 5)), pos_region)
        diff_neg = torch.logical_and((_logits > thred), ~pos_region)
        cc = lcc(diff_pos, diff_neg, point_coords=point_coords, point_labels=point_labels)
        # cc is the region that can be updated by point_logits.
        cc = cc.to(logits.device)
        # Need to replace NaN with point_logits. diff_neg will never lie in nan_mask,
        # only remove unconnected positive region.
        uc_pos_region = torch.logical_and(pos_region, ~cc)
        fill_mask = torch.logical_and(nan_mask, uc_pos_region)
        if fill_mask.any():
            # fill in the mean negative value
            point_logits[fill_mask] = -1
        # replace logits nan value and cc with point_logits
        cc = torch.logical_or(nan_mask, cc).to(logits.dtype)
        logits[mapping_index] *= 1 - cc
        logits[mapping_index] += cc * point_logits
        return logits

    def gaussian_combine(
        self,
        logits: torch.Tensor,
        point_logits: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mapping_index: torch.Tensor,
        radius: int | None = None,
    ):
        """
        Combine point results with auto results using gaussian.

        Args:
            logits: automatic branch results, [B, 1, H, W, D].
            point_logits: point branch results, [B1, 1, H, W, D].
            point_coords: point coordinates, [B1, N, 3].
            point_labels: point labels, [B1, N].
            mapping_index: [B].
            radius: gaussian ball radius.
        """
        if radius is None:
            radius = min(point_logits.shape[-3:]) // 5  # empirical value 5
        weight = 1 - convert_points_to_disc(point_logits.shape[-3:], point_coords, point_labels, radius=radius).sum(
            1, keepdims=True
        )
        weight[weight < 0] = 0
        logits = logits.as_tensor() if isinstance(logits, monai.data.MetaTensor) else logits
        logits[mapping_index] *= weight
        logits[mapping_index] += (1 - weight) * point_logits
        return logits

    def set_auto_grad(self, auto_freeze: bool = False, point_freeze: bool = False):
        """
        Freeze auto-branch or point-branch.

        Args:
            auto_freeze: whether to freeze the auto branch.
            point_freeze: whether to freeze the point branch.
        """
        if auto_freeze != self.auto_freeze:
            if hasattr(self.image_encoder, "set_auto_grad"):
                self.image_encoder.set_auto_grad(auto_freeze=auto_freeze, point_freeze=point_freeze)
            else:
                for param in self.image_encoder.parameters():
                    param.requires_grad = (not auto_freeze) and (not point_freeze)
            for param in self.class_head.parameters():
                param.requires_grad = not auto_freeze
            self.auto_freeze = auto_freeze

        if point_freeze != self.point_freeze:
            if hasattr(self.image_encoder, "set_auto_grad"):
                self.image_encoder.set_auto_grad(auto_freeze=auto_freeze, point_freeze=point_freeze)
            else:
                for param in self.image_encoder.parameters():
                    param.requires_grad = (not auto_freeze) and (not point_freeze)
            for param in self.point_head.parameters():
                param.requires_grad = not point_freeze
            self.point_freeze = point_freeze

    def forward(
        self,
        input_images: torch.Tensor,
        point_coords: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        class_vector: torch.Tensor | None = None,
        prompt_class: torch.Tensor | None = None,
        patch_coords: Sequence[slice] | None = None,
        labels: torch.Tensor | None = None,
        label_set: Sequence[int] | None = None,
        prev_mask: torch.Tensor | None = None,
        radius: int | None = None,
        val_point_sampler: Callable | None = None,
        **kwargs,
    ):
        """
        The forward function for VISTA3D. We only support single patch in training and inference.
        One exception is allowing sliding window batch size > 1 for automatic segmentation only case.
        B represents number of objects, N represents number of points for each objects.

        Args:
            input_images: [1, 1, H, W, D]
            point_coords: [B, N, 3]
            point_labels: [B, N], -1 represents padding. 0/1 means negative/positive points for regular class.
                2/3 means negative/postive ponits for special supported class like tumor.
            class_vector: [B, 1], the global class index
            prompt_class: [B, 1], the global class index. This value is associated with point_coords to identify if
                the points are for zero-shot or supported class. When class_vector and point_coords are both
                provided, prompt_class is the same as class_vector. For prompt_class[b] > 512, point_coords[b]
                will be considered novel class.
            patch_coords: a sequence of the python slice objects representing the patch coordinates during sliding window inference.
                This value is passed from sliding_window_inferer. This is an indicator for training phase or validation phase.
            labels: [1, 1, H, W, D], the groundtruth label tensor, only used for point-only evaluation
            label_set: the label index matching the indexes in labels. If labels are mapped to global index using RelabelID,
                this label_set should be global mapped index. If labels are not mapped to global index, e.g. in zero-shot
                evaluation, this label_set should be the original index.
            prev_mask: [B, N, H_fullsize, W_fullsize, D_fullsize].
                This is the transposed raw output from sliding_window_inferer before any postprocessing.
                When user click points to perform auto-results correction, this can be the auto-results.
            radius: single float value controling the gaussian blur when combining point and auto results.
                The gaussian combine is not used in VISTA3D training but might be useful for finetuning purposes.
            val_point_sampler: function used to sample points from labels. This is only used for point-only evaluation.

        """
        image_size = input_images.shape[-3:]
        device = input_images.device
        if point_coords is None and class_vector is None:
            return self.NINF_VALUE + torch.zeros([1, 1, *image_size], device=device)

        bs = self.get_foreground_class_count(class_vector, point_coords)
        if patch_coords is not None:
            # if during validation and perform enable based point-validation.
            if labels is not None and label_set is not None:
                # if labels is not None, sample from labels for each patch.
                if val_point_sampler is None:
                    # TODO: think about how to refactor this part.
                    val_point_sampler = self.sample_points_patch_val
                point_coords, point_labels, prompt_class = val_point_sampler(labels, patch_coords, label_set)
                if prompt_class[0].item() == 0:  # type: ignore
                    point_labels[0] = -1  # type: ignore
                labels, prev_mask = None, None
            elif point_coords is not None:
                # If not performing patch-based point only validation, use user provided click points for inference.
                # the point clicks is in original image space, convert it to current patch-coordinate space.
                point_coords, point_labels = self.update_point_to_patch(patch_coords, point_coords, point_labels)  # type: ignore

        if point_coords is not None and point_labels is not None:
            # remove points that used for padding purposes (point_label = -1)
            mapping_index = ((point_labels != -1).sum(1) > 0).to(torch.bool)
            if mapping_index.any():
                point_coords = point_coords[mapping_index]
                point_labels = point_labels[mapping_index]
                if prompt_class is not None:
                    prompt_class = prompt_class[mapping_index]
            else:
                if self.auto_freeze or (class_vector is None and patch_coords is None):
                    # if auto_freeze, point prompt must exist to allow loss backward
                    # in training, class_vector and point cannot both be None due to loss.backward()
                    mapping_index.fill_(True)
                else:
                    point_coords, point_labels = None, None

        if point_coords is None and class_vector is None:
            return self.NINF_VALUE + torch.zeros([bs, 1, *image_size], device=device)

        if self.image_embeddings is not None and kwargs.get("keep_cache", False) and class_vector is None:
            out, out_auto = self.image_embeddings, None
        else:
            out, out_auto = self.image_encoder(
                input_images, with_point=point_coords is not None, with_label=class_vector is not None
            )
        # release memory
        input_images = None  # type: ignore

        # force releasing memories that set to None
        torch.cuda.empty_cache()
        if class_vector is not None:
            logits, _ = self.class_head(out_auto, class_vector)
            if point_coords is not None:
                point_logits = self.point_head(out, point_coords, point_labels, class_vector=prompt_class)
                if patch_coords is None:
                    logits = self.gaussian_combine(
                        logits, point_logits, point_coords, point_labels, mapping_index, radius  # type: ignore
                    )
                else:
                    # during validation use largest component
                    logits = self.connected_components_combine(
                        logits, point_logits, point_coords, point_labels, mapping_index  # type: ignore
                    )
        else:
            logits = self.NINF_VALUE + torch.zeros([bs, 1, *image_size], device=device, dtype=out.dtype)
            logits[mapping_index] = self.point_head(out, point_coords, point_labels, class_vector=prompt_class)
            if prev_mask is not None and patch_coords is not None:
                logits = self.connected_components_combine(
                    prev_mask[patch_coords].transpose(1, 0).to(logits.device),
                    logits[mapping_index],
                    point_coords,  # type: ignore
                    point_labels,  # type: ignore
                    mapping_index,
                )

        if kwargs.get("keep_cache", False) and class_vector is None:
            self.image_embeddings = out.detach()
        return logits


class PointMappingSAM(nn.Module):
    def __init__(self, feature_size: int, max_prompt: int = 32, n_classes: int = 512, last_supported: int = 132):
        """Interactive point head used for VISTA3D.
        Adapted from segment anything:
        `https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py`.

        Args:
            feature_size: feature channel from encoder.
            max_prompt: max prompt number in each forward iteration.
            n_classes: number of classes the model can potentially support. This is the maximum number of class embeddings.
            last_supported: number of classes the model support, this value should match the trained model weights.
        """
        super().__init__()
        transformer_dim = feature_size
        self.max_prompt = max_prompt
        self.feat_downsample = nn.Sequential(
            nn.Conv3d(in_channels=feature_size, out_channels=feature_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.GELU(),
            nn.Conv3d(in_channels=feature_size, out_channels=transformer_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(feature_size),
        )

        self.mask_downsample = nn.Conv3d(in_channels=2, out_channels=2, kernel_size=3, stride=2, padding=1)

        self.transformer = TwoWayTransformer(depth=2, embedding_dim=transformer_dim, mlp_dim=512, num_heads=4)
        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, transformer_dim), nn.Embedding(1, transformer_dim)])
        self.not_a_point_embed = nn.Embedding(1, transformer_dim)
        self.special_class_embed = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, transformer_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(transformer_dim),
            nn.GELU(),
            nn.Conv3d(transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1),
        )

        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim, 3)
        # class embedding
        self.n_classes = n_classes
        self.last_supported = last_supported
        self.class_embeddings = nn.Embedding(n_classes, feature_size)
        self.zeroshot_embed = nn.Embedding(1, transformer_dim)
        self.supported_embed = nn.Embedding(1, transformer_dim)

    def forward(
        self,
        out: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        class_vector: torch.Tensor | None = None,
    ):
        """Args:
        out: feature from encoder, [1, C, H, W, C]
        point_coords: point coordinates, [B, N, 3]
        point_labels: point labels, [B, N]
        class_vector: class prompts, [B]
        """
        # downsample out
        out_low = self.feat_downsample(out)
        out_shape = tuple(out.shape[-3:])
        # release memory
        out = None  # type: ignore
        torch.cuda.empty_cache()
        # embed points
        points = point_coords + 0.5  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(points, out_shape)  # type: ignore
        point_embedding[point_labels == -1] = 0.0
        point_embedding[point_labels == -1] += self.not_a_point_embed.weight
        point_embedding[point_labels == 0] += self.point_embeddings[0].weight
        point_embedding[point_labels == 1] += self.point_embeddings[1].weight
        point_embedding[point_labels == 2] += self.point_embeddings[0].weight + self.special_class_embed.weight
        point_embedding[point_labels == 3] += self.point_embeddings[1].weight + self.special_class_embed.weight
        output_tokens = self.mask_tokens.weight

        output_tokens = output_tokens.unsqueeze(0).expand(point_embedding.size(0), -1, -1)
        if class_vector is None:
            tokens_all = torch.cat(
                (
                    output_tokens,
                    point_embedding,
                    self.supported_embed.weight.unsqueeze(0).expand(point_embedding.size(0), -1, -1),
                ),
                dim=1,
            )
            # tokens_all = torch.cat((output_tokens, point_embedding), dim=1)
        else:
            class_embeddings = []
            for i in class_vector:
                if i > self.last_supported:
                    class_embeddings.append(self.zeroshot_embed.weight)
                else:
                    class_embeddings.append(self.supported_embed.weight)
            tokens_all = torch.cat((output_tokens, point_embedding, torch.stack(class_embeddings)), dim=1)
        # cross attention
        masks = []
        max_prompt = self.max_prompt
        for i in range(int(np.ceil(tokens_all.shape[0] / max_prompt))):
            # remove variables in previous for loops to save peak memory for self.transformer
            src, upscaled_embedding, hyper_in = None, None, None
            torch.cuda.empty_cache()
            idx = (i * max_prompt, min((i + 1) * max_prompt, tokens_all.shape[0]))
            tokens = tokens_all[idx[0] : idx[1]]
            src = torch.repeat_interleave(out_low, tokens.shape[0], dim=0)
            pos_src = torch.repeat_interleave(self.pe_layer(out_low.shape[-3:]).unsqueeze(0), tokens.shape[0], dim=0)
            b, c, h, w, d = src.shape
            hs, src = self.transformer(src, pos_src, tokens)
            mask_tokens_out = hs[:, :1, :]
            hyper_in = self.output_hypernetworks_mlps(mask_tokens_out)
            src = src.transpose(1, 2).view(b, c, h, w, d)  # type: ignore
            upscaled_embedding = self.output_upscaling(src)
            b, c, h, w, d = upscaled_embedding.shape
            mask = hyper_in @ upscaled_embedding.view(b, c, h * w * d)
            masks.append(mask.view(-1, 1, h, w, d))

        return torch.vstack(masks)


class ClassMappingClassify(nn.Module):
    """Class head that performs automatic segmentation based on class vector."""

    def __init__(self, n_classes: int, feature_size: int, use_mlp: bool = True):
        """Args:
        n_classes: maximum number of class embedding.
        feature_size: class embedding size.
        use_mlp: use mlp to further map class embedding.
        """
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.InstanceNorm1d(1),
                nn.GELU(),
                nn.Linear(feature_size, feature_size),
            )
        self.class_embeddings = nn.Embedding(n_classes, feature_size)
        self.image_post_mapping = nn.Sequential(
            UnetrBasicBlock(
                spatial_dims=3,
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name="instance",
                res_block=True,
            ),
            UnetrBasicBlock(
                spatial_dims=3,
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name="instance",
                res_block=True,
            ),
        )

    def forward(self, src: torch.Tensor, class_vector: torch.Tensor):
        b, c, h, w, d = src.shape
        src = self.image_post_mapping(src)
        class_embedding = self.class_embeddings(class_vector)
        if self.use_mlp:
            class_embedding = self.mlp(class_embedding)
        # [b,1,feat] @ [1,feat,dim], batch dimension become class_embedding batch dimension.
        masks = []
        for i in range(b):
            mask = class_embedding @ src[[i]].view(1, c, h * w * d)
            masks.append(mask.view(-1, 1, h, w, d))

        return torch.cat(masks, 1), class_embedding


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: tuple | str = "relu",
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.
        Adapted from `https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.py`.

        Args:
            depth: number of layers in the transformer.
            embedding_dim: the channel dimension for the input embeddings.
            num_heads: the number of heads for multihead attention. Must divide embedding_dim.
            mlp_dim: the channel dimension internal to the MLP block.
            activation: the activation to use in the MLP block.
            attention_downsample_rate: the rate at which to downsample the image before projecting.
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self, image_embedding: torch.Tensor, image_pe: torch.Tensor, point_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embedding: image to attend to. Should be shape
                B x embedding_dim x h x w for any h and w.
            image_pe: the positional encoding to add to the image. Must
                have the same shape as image_embedding.
            point_embedding: the embedding to add to the query points.
                Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
            torch.Tensor: the processed point_embedding.
            torch.Tensor: the processed image_embedding.
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: tuple | str = "relu",
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.
        Adapted from `https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.py`.

        Args:
            embedding_dim: the channel dimension of the embeddings.
            num_heads: the number of heads in the attention layers.
            mlp_dim: the hidden dimension of the mlp block.
            activation: the activation of the mlp block.
            skip_first_layer_pe: skip the PE on the first layer.
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(hidden_size=embedding_dim, mlp_dim=mlp_dim, act=activation, dropout_mode="vista3d")
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    Adapted from `https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.py`.

    Args:
        embedding_dim: the channel dimension of the embeddings.
        num_heads: the number of heads in the attention layers.
        downsample_rate: the rate at which to downsample the image before projecting.
    """

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if not self.internal_dim % num_heads == 0:
            raise ValueError("num_heads must divide embedding_dim.")

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        # B x N_heads x N_tokens x C_per_head
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        # B x N_tokens x C
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    Adapted from `https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py`.

    Args:
        num_pos_feats: the number of positional encoding features.
        scale: the scale of the positional encoding.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((3, num_pos_feats)))

    def _pe_encoding(self, coords: torch.torch.Tensor) -> torch.torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        # [bs=1,N=2,2] @ [2,128]
        # [bs=1, N=2, 128]
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        # [bs=1, N=2, 128+128=256]
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w, d = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w, d), device=device, dtype=torch.float32)
        x_embed = grid.cumsum(dim=0) - 0.5
        y_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        x_embed = x_embed / h
        y_embed = y_embed / w
        z_embed = z_embed / d
        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        # C x H x W
        return pe.permute(3, 0, 1, 2)

    def forward_with_coords(
        self, coords_input: torch.torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[0]
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        # B x N x C
        return self._pe_encoding(coords.to(torch.float))


class MLP(nn.Module):
    """
    Multi-layer perceptron. This class is only used for `PointMappingSAM`.
    Adapted from `https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py`.

    Args:
        input_dim: the input dimension.
        hidden_dim: the hidden dimension.
        output_dim: the output dimension.
        num_layers: the number of layers.
        sigmoid_output: whether to apply a sigmoid activation to the output.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
