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
# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
# which has the following license...
# https://github.com/pytorch/vision/blob/main/LICENSE

# BSD 3-Clause License

# Copyright (c) Soumith Chintala 2016,
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

"""
Part of this script is adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
"""

import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.utils.anchor_utils import AnchorGenerator
from monai.apps.detection.utils.ATSS_matcher import ATSSMatcher
from monai.apps.detection.utils.box_coder import BoxCoder
from monai.apps.detection.utils.box_selector import BoxSelector
from monai.apps.detection.utils.detector_utils import check_training_targets, preprocess_images
from monai.apps.detection.utils.hard_negative_sampler import HardNegativeSampler
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_, predict_with_inferer
from monai.data.box_utils import box_iou
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import resnet
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple_rep, optional_import

BalancedPositiveNegativeSampler, _ = optional_import(
    "torchvision.models.detection._utils", name="BalancedPositiveNegativeSampler"
)
Matcher, _ = optional_import("torchvision.models.detection._utils", name="Matcher")


class RetinaNetDetector(nn.Module):
    """
    Retinanet detector, expandable to other one stage anchor based box detectors in the future.
    An example of construction can found in the source code of
     :func:`~monai.apps.detection.networks.retinanet_detector.retinanet_resnet50_fpn_detector` .

    The input to the model is expected to be a list of tensors, each of shape (C, H, W) or  (C, H, W, D),
    one for each image, and should be in 0-1 range. Different images can have different sizes.
    Or it can also be a Tensor sized (B, C, H, W) or  (B, C, H, W, D). In this case, all images have same size.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]`` or ``FloatTensor[N, 6]``): the ground-truth boxes in ``StandardMode``, i.e.,
            ``[xmin, ymin, xmax, ymax]`` or ``[xmin, ymin, zmin, xmax, ymax, zmax]`` format,
            with ``0 <= xmin < xmax <= H``, ``0 <= ymin < ymax <= W``, ``0 <= zmin < zmax <= D``.
        - labels: the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.
    When save the model, only self.network contains trainable parameters and needs to be saved.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]`` or ``FloatTensor[N, 6]``): the ground-truth boxes in ``StandardMode``, i.e.,
            ``[xmin, ymin, xmax, ymax]`` or ``[xmin, ymin, zmin, xmax, ymax, zmax]`` format,
            with ``0 <= xmin < xmax <= H``, ``0 <= ymin < ymax <= W``, ``0 <= zmin < zmax <= D``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - labels_scores (Tensor[N]): the scores for each prediction

    Args:
        network: a network that takes an image Tensor sized (B, C, H, W) or (B, C, H, W, D) as input
            and outputs a dictionary Dict[str, List[Tensor]].
        anchor_generator: anchor generator.
        box_overlap_metric: func that compute overlap between two sets of boxes, default is Intersection over Union (IoU).
        debug: whether to print out internal parameters, used for debugging and parameter tuning.

    Notes:
        Input argument ``network`` can be a monai.apps.detection.networks.retinanet_network.RetinaNet(*) object,
        but any network that meets the following rules is a valid input ``network``.

        1. It should have attributes including spatial_dims, num_classes, cls_key, box_reg_key, num_anchors, size_divisible.

            - spatial_dims (int) is the spatial dimension of the network, we support both 2D and 3D.
            - num_classes (int) is the number of classes, excluding the background.
            - size_divisible (int or Sequene[int]) is the expection on the input image shape.
              The network needs the input spatial_size to be divisible by size_divisible, length should be 2 or 3.
            - cls_key (str) is the key to represent classification in the output dict.
            - box_reg_key (str) is the key to represent box regression in the output dict.
            - num_anchors (int) is the number of anchor shapes at each location. it should equal to
              ``self.anchor_generator.num_anchors_per_location()[0]``.

        2. Its input should be an image Tensor sized (B, C, H, W) or (B, C, H, W, D).

        3. About its output ``head_outputs``:

            - It should be a dictionary with at least two keys:
              ``network.cls_key`` and ``network.box_reg_key``.
            - ``head_outputs[network.cls_key]`` should be List[Tensor] or Tensor. Each Tensor represents
              classification logits map at one resolution level,
              sized (B, num_classes*num_anchors, H_i, W_i) or (B, num_classes*num_anchors, H_i, W_i, D_i).
            - ``head_outputs[network.box_reg_key]`` should be List[Tensor] or Tensor. Each Tensor represents
              box regression map at one resolution level,
              sized (B, 2*spatial_dims*num_anchors, H_i, W_i)or (B, 2*spatial_dims*num_anchors, H_i, W_i, D_i).
            - ``len(head_outputs[network.cls_key]) == len(head_outputs[network.box_reg_key])``.

    Example:
        .. code-block:: python

            # define a naive network
            import torch
            class NaiveNet(torch.nn.Module):
                def __init__(self, spatial_dims: int, num_classes: int):
                    super().__init__()
                    self.spatial_dims = spatial_dims
                    self.num_classes = num_classes
                    self.size_divisible = 2
                    self.cls_key = "cls"
                    self.box_reg_key = "box_reg"
                    self.num_anchors = 1
                def forward(self, images: torch.Tensor):
                    spatial_size = images.shape[-self.spatial_dims:]
                    out_spatial_size = tuple(s//self.size_divisible for s in spatial_size)  # half size of input
                    out_cls_shape = (images.shape[0],self.num_classes*self.num_anchors) + out_spatial_size
                    out_box_reg_shape = (images.shape[0],2*self.spatial_dims*self.num_anchors) + out_spatial_size
                    return {self.cls_key: [torch.randn(out_cls_shape)], self.box_reg_key: [torch.randn(out_box_reg_shape)]}

            # create a RetinaNetDetector detector
            spatial_dims = 3
            num_classes = 5
            anchor_generator = monai.apps.detection.utils.anchor_utils.AnchorGeneratorWithAnchorShape(
                feature_map_scales=(1, ), base_anchor_shapes=((8,) * spatial_dims)
            )
            net = NaiveNet(spatial_dims, num_classes)
            detector = RetinaNetDetector(net, anchor_generator)

            # only detector.network may contain trainable parameters.
            optimizer = torch.optim.SGD(
                detector.network.parameters(),
                1e-3,
                momentum=0.9,
                weight_decay=3e-5,
                nesterov=True,
            )
            torch.save(detector.network.state_dict(), 'model.pt')  # save model
            detector.network.load_state_dict(torch.load('model.pt'))  # load model
    """

    def __init__(
        self, network, anchor_generator: AnchorGenerator, box_overlap_metric: Callable = box_iou, debug: bool = False
    ):
        super().__init__()

        if not all(
            hasattr(network, attr)
            for attr in ["spatial_dims", "num_classes", "cls_key", "box_reg_key", "num_anchors", "size_divisible"]
        ):
            raise AttributeError(
                "network should have attributes, including: "
                "'spatial_dims', 'num_classes', 'cls_key', 'box_reg_key', 'num_anchors', 'size_divisible'."
            )

        self.network = network
        self.spatial_dims = self.network.spatial_dims
        self.num_classes = self.network.num_classes
        self.size_divisible = ensure_tuple_rep(self.network.size_divisible, self.spatial_dims)
        # keys for the network output
        self.cls_key = self.network.cls_key
        self.box_reg_key = self.network.box_reg_key

        # check if anchor_generator matches with network
        self.anchor_generator = anchor_generator

        self.num_anchors_per_loc = self.anchor_generator.num_anchors_per_location()[0]
        if self.num_anchors_per_loc != self.network.num_anchors:
            raise ValueError(
                f"Number of feature map channels ({self.network.num_anchors}) "
                f"should match with number of anchors at each location ({self.num_anchors_per_loc})."
            )
        # if new coming input images has same shape with
        # self.previous_image_shape, there is no need to generate new anchors.
        self.anchors: Union[List[Tensor], None] = None
        self.previous_image_shape: Union[Any, None] = None

        self.box_overlap_metric = box_overlap_metric
        self.debug = debug

        # default setting for training
        self.fg_bg_sampler: Union[Any, None] = None
        self.set_cls_loss(torch.nn.BCEWithLogitsLoss(reduction="mean"))  # classification loss
        self.set_box_regression_loss(
            torch.nn.SmoothL1Loss(beta=1.0 / 9, reduction="mean"), encode_gt=True, decode_pred=False
        )  # box regression loss

        # default setting for both training and inference
        # can be updated by self.set_box_coder_weights(*)
        self.box_coder = BoxCoder(weights=(1.0,) * 2 * self.spatial_dims)

        # default keys in the ground truth targets and predicted boxes,
        # can be updated by self.set_target_keys(*)
        self.target_box_key = "boxes"
        self.target_label_key = "labels"
        self.pred_score_key = self.target_label_key + "_scores"  # score key for the detected boxes

        # default setting for inference,
        # can be updated by self.set_sliding_window_inferer(*)
        self.inferer: Union[SlidingWindowInferer, None] = None
        # can be updated by self.set_box_selector_parameters(*),
        self.box_selector = BoxSelector(
            box_overlap_metric=self.box_overlap_metric,
            score_thresh=0.05,
            topk_candidates_per_level=1000,
            nms_thresh=0.5,
            detections_per_img=300,
            apply_sigmoid=True,
        )

    def set_box_coder_weights(self, weights: Tuple[float]):
        """
        Set the weights for box coder.

        Args:
            weights: a list/tuple with length of 2*self.spatial_dims

        """
        if len(weights) != 2 * self.spatial_dims:
            raise ValueError(f"len(weights) should be {2 * self.spatial_dims}, got weights={weights}.")
        self.box_coder = BoxCoder(weights=weights)

    def set_target_keys(self, box_key: str, label_key: str):
        """
        Set keys for the training targets and inference outputs.
        During training, both box_key and label_key should be keys in the targets
        when performing ``self.forward(input_images, targets)``.
        During inference, they will be the keys in the output dict of `self.forward(input_images)``.
        """
        self.target_box_key = box_key
        self.target_label_key = label_key
        self.pred_score_key = label_key + "_scores"

    def set_cls_loss(self, cls_loss: nn.Module) -> None:
        """
        Using for training. Set loss for classification that takes logits as inputs, make sure sigmoid/softmax is built in.

        Args:
            cls_loss: loss module for classification

        Example:
            .. code-block:: python

                detector.set_cls_loss(torch.nn.BCEWithLogitsLoss(reduction="mean"))
                detector.set_cls_loss(FocalLoss(reduction="mean", gamma=2.0))
        """
        self.cls_loss_func = cls_loss

    def set_box_regression_loss(self, box_loss: nn.Module, encode_gt: bool, decode_pred: bool) -> None:
        """
        Using for training. Set loss for box regression.

        Args:
            box_loss: loss module for box regression
            encode_gt: if True, will encode ground truth boxes to target box regression
                before computing the losses. Should be True for L1 loss and False for GIoU loss.
            decode_pred: if True, will decode predicted box regression into predicted boxes
                before computing losses. Should be False for L1 loss and True for GIoU loss.

        Example:
            .. code-block:: python

                detector.set_box_regression_loss(
                    torch.nn.SmoothL1Loss(beta=1.0 / 9, reduction="mean"),
                    encode_gt = True, decode_pred = False
                )
        """
        self.box_loss_func = box_loss
        self.encode_gt = encode_gt
        self.decode_pred = decode_pred

    def set_regular_matcher(self, fg_iou_thresh: float, bg_iou_thresh: float, allow_low_quality_matches=True) -> None:
        """
        Using for training. Set torchvision matcher that matches anchors with ground truth boxes.

        Args:
            fg_iou_thresh: foreground IoU threshold for Matcher, considered as matched if IoU > fg_iou_thresh
            bg_iou_thresh: background IoU threshold for Matcher, considered as not matched if IoU < bg_iou_thresh
        """
        if fg_iou_thresh < bg_iou_thresh:
            raise ValueError(
                "Require fg_iou_thresh >= bg_iou_thresh. "
                f"Got fg_iou_thresh={fg_iou_thresh}, bg_iou_thresh={bg_iou_thresh}."
            )
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)

    def set_atss_matcher(self, num_candidates: int = 4, center_in_gt: bool = False) -> None:
        """
        Using for training. Set ATSS matcher that matches anchors with ground truth boxes

        Args:
            num_candidates: number of positions to select candidates from.
                Smaller value will result in a higher matcher threshold and less matched candidates.
            center_in_gt: If False (default), matched anchor center points do not need
                to lie withing the ground truth box. Recommend False for small objects.
                If True, will result in a strict matcher and less matched candidates.
        """
        self.proposal_matcher = ATSSMatcher(num_candidates, self.box_overlap_metric, center_in_gt, debug=self.debug)

    def set_hard_negative_sampler(
        self, batch_size_per_image: int, positive_fraction: float, min_neg: int = 1, pool_size: float = 10
    ):
        """
        Using for training. Set hard negative sampler that samples part of the anchors for training.

        HardNegativeSampler is used to suppress false positive rate in classification tasks.
        During training, it select negative samples with high prediction scores.

        Args:
            batch_size_per_image: number of elements to be selected per image
            positive_fraction: percentage of positive elements in the selected samples
            min_neg: minimum number of negative samples to select if possible.
            pool_size: when we need ``num_neg`` hard negative samples, they will be randomly selected from
                ``num_neg * pool_size`` negative samples with the highest prediction scores.
                Larger ``pool_size`` gives more randomness, yet selects negative samples that are less 'hard',
                i.e., negative samples with lower prediction scores.
        """
        self.fg_bg_sampler = HardNegativeSampler(
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            min_neg=min_neg,
            pool_size=pool_size,
        )

    def set_balanced_sampler(self, batch_size_per_image: int, positive_fraction: float):
        """
        Using for training. Set torchvision balanced sampler that samples part of the anchors for training.

        Args:
            batch_size_per_image: number of elements to be selected per image
            positive_fraction: percentage of positive elements per batch

        """
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image=batch_size_per_image, positive_fraction=positive_fraction
        )

    def set_sliding_window_inferer(
        self,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int = 1,
        overlap: float = 0.5,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
        progress: bool = False,
        cache_roi_weight_map: bool = False,
    ):
        """
        Define sliding window inferer and store it to self.inferer.
        """
        self.inferer = SlidingWindowInferer(
            roi_size,
            sw_batch_size,
            overlap,
            mode,
            sigma_scale,
            padding_mode,
            cval,
            sw_device,
            device,
            progress,
            cache_roi_weight_map,
        )

    def set_box_selector_parameters(
        self,
        score_thresh: float = 0.05,
        topk_candidates_per_level: int = 1000,
        nms_thresh: float = 0.5,
        detections_per_img: int = 300,
        apply_sigmoid: bool = True,
    ):
        """
        Using for inference. Set the parameters that are used for box selection during inference.
        The box selection is performed with the following steps:

        #. For each level, discard boxes with scores less than self.score_thresh.
        #. For each level, keep boxes with top self.topk_candidates_per_level scores.
        #. For the whole image, perform non-maximum suppression (NMS) on boxes, with overapping threshold nms_thresh.
        #. For the whole image, keep boxes with top self.detections_per_img scores.

        Args:
            score_thresh: no box with scores less than score_thresh will be kept
            topk_candidates_per_level: max number of boxes to keep for each level
            nms_thresh: box overlapping threshold for NMS
            detections_per_img: max number of boxes to keep for each image
        """

        self.box_selector = BoxSelector(
            box_overlap_metric=self.box_overlap_metric,
            apply_sigmoid=apply_sigmoid,
            score_thresh=score_thresh,
            topk_candidates_per_level=topk_candidates_per_level,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )

    def forward(
        self,
        input_images: Union[List[Tensor], Tensor],
        targets: Union[List[Dict[str, Tensor]], None] = None,
        use_inferer: bool = False,
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Returns a dict of losses during training, or a list predicted dict of boxes and labels during inference.

        Args:
            input_images: The input to the model is expected to be a list of tensors, each of shape (C, H, W) or  (C, H, W, D),
                one for each image, and should be in 0-1 range. Different images can have different sizes.
                Or it can also be a Tensor sized (B, C, H, W) or  (B, C, H, W, D). In this case, all images have same size.
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image (optional).
            use_inferer: whether to use self.inferer, a sliding window inferer, to do the inference.
                If False, will simply forward the network.
                If True, will use self.inferer, and requires
                ``self.set_sliding_window_inferer(*args)`` to have been called before.

        Return:
            If training mode, will return a dict with at least two keys,
            including self.cls_key and self.box_reg_key, representing classification loss and box regression loss.

            If evaluation mode, will return a list of detection results.
            Each element corresponds to an images in ``input_images``, is a dict with at least three keys,
            including self.target_box_key, self.target_label_key, self.pred_score_key,
            representing predicted boxes, classification labels, and classification scores.

        """
        # 1. Check if input arguments are valid
        if self.training:
            check_training_targets(input_images, targets, self.spatial_dims, self.target_label_key, self.target_box_key)
            self.check_detector_training_components()

        # 2. Pad list of images to a single Tensor `images` with spatial size divisible by self.size_divisible.
        # image_sizes stores the original spatial_size of each image before padding.
        images, image_sizes = preprocess_images(input_images, self.spatial_dims, self.size_divisible)

        # 3. Generate network outputs. Use inferer only in evaluation mode.
        if self.training or (not use_inferer):
            head_outputs = self.network(images)
            ensure_dict_value_to_list_(head_outputs)  # ensure head_outputs is Dict[str, List[Tensor]]
        else:
            if self.inferer is None:
                raise ValueError(
                    "`self.inferer` is not defined." "Please refer to function self.set_sliding_window_inferer(*)."
                )
            head_outputs = predict_with_inferer(
                images, self.network, keys=[self.cls_key, self.box_reg_key], inferer=self.inferer
            )

        # 4. Generate anchors and store it in self.anchors: List[Tensor]
        self.generate_anchors(images, head_outputs)
        # num_anchor_locs_per_level: List[int], list of HW or HWD for each level
        num_anchor_locs_per_level = [x.shape[2:].numel() for x in head_outputs[self.cls_key]]

        # 5. Reshape and concatenate head_outputs values from List[Tensor] to Tensor
        # head_outputs, originally being Dict[str, List[Tensor]], will be reshaped to Dict[str, Tensor]
        for key in [self.cls_key, self.box_reg_key]:
            # reshape to Tensor sized(B, sum(HWA), self.num_classes) for self.cls_key
            # or (B, sum(HWA), 2* self.spatial_dims) for self.box_reg_key
            # A = self.num_anchors_per_loc
            head_outputs[key] = self.reshape_maps(head_outputs[key])

        # 6(1). If during training, return losses
        if self.training:
            losses = self.compute_loss(head_outputs, targets, self.anchors, num_anchor_locs_per_level)  # type: ignore
            return losses

        # 6(2). If during inference, return detection results
        detections = self.postprocess_detections(
            head_outputs, self.anchors, image_sizes, num_anchor_locs_per_level  # type: ignore
        )
        return detections

    def check_detector_training_components(self):
        if not hasattr(self, "proposal_matcher"):
            raise AttributeError(
                "Matcher is not set. Please refer to self.set_regular_matcher(*) or self.set_atss_matcher(*)."
            )
        if self.fg_bg_sampler is None and self.debug:
            warnings.warn(
                "No balanced sampler is used. Negative samples are likely to "
                "be much more than positive samples. Please set balanced samplers with self.set_balanced_sampler(*) "
                "or self.set_hard_negative_sampler(*), "
                "or set classification loss function as Focal loss with self.set_cls_loss(*)"
            )

    def generate_anchors(self, images: Tensor, head_outputs: Dict[str, List[Tensor]]):
        """
        Generate anchors and store it in self.anchors: List[Tensor].
        We generate anchors only when there is no stored anchors,
        or the new coming images has different shape with self.previous_image_shape

        Args:
            images: input images, a (B, C, H, W) or (B, C, H, W, D) Tensor.
            head_outputs: head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
              sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
              sized (B, sum(HW(D)A), 2*self.spatial_dims)
        """
        if (self.anchors is None) or (self.previous_image_shape != images.shape):
            self.anchors = self.anchor_generator(images, head_outputs[self.cls_key])  # List[Tensor], len = batchsize
            self.previous_image_shape = images.shape

    def reshape_maps(self, result_maps: List[Tensor]) -> Tensor:
        """
        Concat network output map list to a single Tensor.
        This function is used in both training and inference.

        Args:
            result_maps: a list of Tensor, each Tensor is a (B, num_channel*A, H, W) or (B, num_channel*A, H, W, D) map.
                A = self.num_anchors_per_loc

        Return:
            reshaped and concatenated result, sized (B, sum(HWA), num_channel) or (B, sum(HWDA), num_channel)
        """
        all_reshaped_result_map = []

        for result_map in result_maps:
            batch_size = result_map.shape[0]
            num_channel = result_map.shape[1] // self.num_anchors_per_loc
            spatial_size = result_map.shape[-self.spatial_dims :]

            # reshaped_result_map will become (B, A, num_channel, H, W) or (B, A, num_channel, H, W, D)
            # A = self.num_anchors_per_loc
            view_shape = (batch_size, -1, num_channel) + spatial_size
            reshaped_result_map = result_map.view(view_shape)

            # permute output to (B, H, W, A, num_channel) or (B, H, W, D, A, num_channel)
            if self.spatial_dims == 2:
                reshaped_result_map = reshaped_result_map.permute(0, 3, 4, 1, 2)
            elif self.spatial_dims == 3:
                reshaped_result_map = reshaped_result_map.permute(0, 3, 4, 5, 1, 2)
            else:
                ValueError("Images can only be 2D or 3D.")

            # reshaped_result_map will become (B, HWA, num_channel) or (B, HWDA, num_channel)
            reshaped_result_map = reshaped_result_map.reshape(batch_size, -1, num_channel)

            if torch.isnan(reshaped_result_map).any() or torch.isinf(reshaped_result_map).any():
                raise ValueError("Concatenated result is NaN or Inf.")

            all_reshaped_result_map.append(reshaped_result_map)

        return torch.cat(all_reshaped_result_map, dim=1)

    def postprocess_detections(
        self,
        head_outputs_reshape: Dict[str, Tensor],
        anchors: List[Tensor],
        image_sizes: List[List[int]],
        num_anchor_locs_per_level: Sequence[int],
        need_sigmoid: bool = True,
    ) -> List[Dict[str, Tensor]]:
        """
        Postprocessing to generate detection result from classification logits and box regression.
        Use self.box_selector to select the final outut boxes for each image.

        Args:
            head_outputs_reshape: reshaped head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
              sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
              sized (B, sum(HW(D)A), 2*self.spatial_dims)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.

        Return:
            a list of dict, each dict scorresponds to detection result on image.
        """

        # recover level sizes, HWA or HWDA for each level
        num_anchors_per_level = [
            num_anchor_locs * self.num_anchors_per_loc for num_anchor_locs in num_anchor_locs_per_level
        ]

        # split outputs per level
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k in head_outputs_reshape:
            split_head_outputs[k] = list(head_outputs_reshape[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]  # List[List[Tensor]]

        class_logits = split_head_outputs[self.cls_key]  # List[Tensor], each sized (B, HWA, self.num_classes)
        box_regression = split_head_outputs[self.box_reg_key]  # List[Tensor], each sized (B, HWA, 2*spatial_dims)
        compute_dtype = class_logits[0].dtype

        num_images = len(image_sizes)  # B

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [
                br[index] for br in box_regression
            ]  # List[Tensor], each sized (HWA, 2*spatial_dims)
            logits_per_image = [cl[index] for cl in class_logits]  # List[Tensor], each sized (HWA, self.num_classes)
            anchors_per_image, img_spatial_size = split_anchors[index], image_sizes[index]
            # decode box regression into boxes
            boxes_per_image = [
                self.box_coder.decode_single(b.to(torch.float32), a).to(compute_dtype)
                for b, a in zip(box_regression_per_image, anchors_per_image)
            ]  # List[Tensor], each sized (HWA, 2*spatial_dims)

            selected_boxes, selected_scores, selected_labels = self.box_selector.select_boxes_per_image(
                boxes_per_image, logits_per_image, img_spatial_size
            )

            detections.append(
                {
                    self.target_box_key: selected_boxes,  # Tensor, sized (N, 2*spatial_dims)
                    self.pred_score_key: selected_scores,  # Tensor, sized (N, )
                    self.target_label_key: selected_labels,  # Tensor, sized (N, )
                }
            )

        return detections

    def compute_loss(
        self,
        head_outputs_reshape: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        anchors: List[Tensor],
        num_anchor_locs_per_level: Sequence[int],
    ) -> Dict[str, Tensor]:
        """
        Compute losses.

        Args:
            head_outputs_reshape: reshaped head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
              sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
              sized (B, sum(HW(D)A), 2*self.spatial_dims)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.

        Return:
            a dict of several kinds of losses.
        """
        matched_idxs = self.compute_anchor_matched_idxs(anchors, targets, num_anchor_locs_per_level)
        losses_cls = self.compute_cls_loss(head_outputs_reshape[self.cls_key], targets, matched_idxs)
        losses_box_regression = self.compute_box_loss(
            head_outputs_reshape[self.box_reg_key], targets, anchors, matched_idxs
        )
        return {self.cls_key: losses_cls, self.box_reg_key: losses_box_regression}

    def compute_anchor_matched_idxs(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]], num_anchor_locs_per_level: Sequence[int]
    ) -> List[Tensor]:
        """
        Compute the matched indices between anchors and ground truth (gt) boxes in targets.
        output[k][i] represents the matched gt index for anchor[i] in image k.
        Suppose there are M gt boxes for image k. The range of it output[k][i] value is [-2, -1, 0, ..., M-1].
        [0, M - 1] indicates this anchor is matched with a gt box,
        while a negative value indicating that it is not matched.

        Args:
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            num_anchor_locs_per_level: each element represents HW or HWD at this level.


        Return:
            a list of matched index `matched_idxs_per_image` (Tensor[int64]), Tensor sized (sum(HWA),) or (sum(HWDA),).
            Suppose there are M gt boxes. `matched_idxs_per_image[i]` is a matched gt index in [0, M - 1]
            or a negative value indicating that anchor i could not be matched.
            BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2
        """
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # anchors_per_image: Tensor, targets_per_image: Dice[str, Tensor]
            if targets_per_image[self.target_box_key].numel() == 0:
                # if no GT boxes
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            # matched_idxs_per_image (Tensor[int64]): Tensor sized (sum(HWA),) or (sum(HWDA),)
            # Suppose there are M gt boxes. matched_idxs_per_image[i] is a matched gt index in [0, M - 1]
            # or a negative value indicating that anchor i could not be matched.
            # BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2
            if isinstance(self.proposal_matcher, Matcher):
                # if torcvision matcher
                match_quality_matrix = self.box_overlap_metric(
                    targets_per_image[self.target_box_key].to(anchors_per_image.device), anchors_per_image
                )
                matched_idxs_per_image = self.proposal_matcher(match_quality_matrix)
            elif isinstance(self.proposal_matcher, ATSSMatcher):
                # if monai ATSS matcher
                match_quality_matrix, matched_idxs_per_image = self.proposal_matcher(
                    targets_per_image[self.target_box_key].to(anchors_per_image.device),
                    anchors_per_image,
                    num_anchor_locs_per_level,
                    self.num_anchors_per_loc,
                )
            else:
                raise NotImplementedError(
                    "Currently support torchvision Matcher and monai ATSS matcher. Other types of matcher not supported. "
                    "Please override self.compute_anchor_matched_idxs(*) for your own matcher."
                )

            if self.debug:
                print(f"Max box overlap between anchors and gt boxes: {torch.max(match_quality_matrix,dim=1)[0]}.")

            if torch.max(matched_idxs_per_image) < 0:
                warnings.warn(
                    f"No anchor is matched with GT boxes. Please adjust matcher setting, anchor setting,"
                    " or the network setting to change zoom scale between network output and input images."
                    f"GT boxes are {targets_per_image[self.target_box_key]}."
                )

            matched_idxs.append(matched_idxs_per_image)
        return matched_idxs

    def compute_cls_loss(
        self, cls_logits: Tensor, targets: List[Dict[str, Tensor]], matched_idxs: List[Tensor]
    ) -> Tensor:
        """
        Compute classification losses.

        Args:
            cls_logits: classification logits, sized (B, sum(HW(D)A), self.num_classes)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            matched_idxs: a list of matched index. each element is sized (sum(HWA),) or  (sum(HWDA),)

        Return:
            classification losses.
        """
        total_cls_logits_list = []
        total_gt_classes_target_list = []
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # for each image, get training samples
            sampled_cls_logits_per_image, sampled_gt_classes_target = self.get_cls_train_sample_per_image(
                cls_logits_per_image, targets_per_image, matched_idxs_per_image
            )
            total_cls_logits_list.append(sampled_cls_logits_per_image)
            total_gt_classes_target_list.append(sampled_gt_classes_target)

        total_cls_logits = torch.cat(total_cls_logits_list, dim=0)
        total_gt_classes_target = torch.cat(total_gt_classes_target_list, dim=0)
        losses: Tensor = self.cls_loss_func(total_cls_logits, total_gt_classes_target).to(total_cls_logits.dtype)
        return losses

    def compute_box_loss(
        self,
        box_regression: Tensor,
        targets: List[Dict[str, Tensor]],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Tensor:
        """
        Compute box regression losses.

        Args:
            box_regression: box regression results, sized (B, sum(HWA), 2*self.spatial_dims)
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors: a list of Tensor. Each Tensor represents anchors for each image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.
            matched_idxs: a list of matched index. each element is sized (sum(HWA),) or  (sum(HWDA),)

        Return:
            box regression losses.
        """
        total_box_regression_list = []
        total_target_regression_list = []

        for targets_per_image, box_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, box_regression, anchors, matched_idxs
        ):
            # for each image, get training samples
            decode_box_regression_per_image, matched_gt_boxes_per_image = self.get_box_train_sample_per_image(
                box_regression_per_image, targets_per_image, anchors_per_image, matched_idxs_per_image
            )
            total_box_regression_list.append(decode_box_regression_per_image)
            total_target_regression_list.append(matched_gt_boxes_per_image)

        total_box_regression = torch.cat(total_box_regression_list, dim=0)
        total_target_regression = torch.cat(total_target_regression_list, dim=0)

        if total_box_regression.shape[0] == 0:
            # if there is no training sample.
            losses = torch.tensor(0.0)
            return losses

        losses = self.box_loss_func(total_box_regression, total_target_regression).to(total_box_regression.dtype)

        return losses

    def get_cls_train_sample_per_image(
        self, cls_logits_per_image: Tensor, targets_per_image: Dict[str, Tensor], matched_idxs_per_image: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Get samples from one image for classification losses computation.

        Args:
            cls_logits_per_image: classification logits for one image, (sum(HWA), self.num_classes)
            targets_per_image: a dict with at least two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            matched_idxs_per_image: matched index, Tensor sized (sum(HWA),) or (sum(HWDA),)
                Suppose there are M gt boxes. matched_idxs_per_image[i] is a matched gt index in [0, M - 1]
                or a negative value indicating that anchor i could not be matched.
                BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2

        Return:
            paired predicted and GT samples from one image for classification losses computation
        """

        if torch.isnan(cls_logits_per_image).any() or torch.isinf(cls_logits_per_image).any():
            raise ValueError("NaN or Inf in predicted classification logits.")

        foreground_idxs_per_image = matched_idxs_per_image >= 0

        num_foreground = foreground_idxs_per_image.sum()
        num_gt_box = targets_per_image[self.target_box_key].shape[0]

        if self.debug:
            print(f"Number of positive (matched) anchors: {num_foreground}; Number of GT box: {num_gt_box}.")
            if num_gt_box > 0 and num_foreground < 2 * num_gt_box:
                print(
                    f"Only {num_foreground} anchors are matched with {num_gt_box} GT boxes. "
                    "Please consider adjusting matcher setting, anchor setting,"
                    " or the network setting to change zoom scale between network output and input images."
                )

        # create the target classification with one-hot encoding
        gt_classes_target = torch.zeros_like(cls_logits_per_image)  # (sum(HW(D)A), self.num_classes)
        gt_classes_target[
            foreground_idxs_per_image,  # fg anchor idx in
            targets_per_image[self.target_label_key][
                matched_idxs_per_image[foreground_idxs_per_image]
            ],  # fg class label
        ] = 1.0

        if self.fg_bg_sampler is None:
            # if no balanced sampling
            valid_idxs_per_image = matched_idxs_per_image != self.proposal_matcher.BETWEEN_THRESHOLDS
        else:
            # The input of fg_bg_sampler: list of tensors containing -1, 0 or positive values.
            # Each tensor corresponds to a specific image.
            # -1 values are ignored, 0 are considered as negatives and > 0 as positives.

            # matched_idxs_per_image (Tensor[int64]): an N tensor where N[i] is a matched gt in
            # [0, M - 1] or a negative value indicating that prediction i could not
            # be matched. BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2
            if isinstance(self.fg_bg_sampler, HardNegativeSampler):
                max_cls_logits_per_image = torch.max(cls_logits_per_image.to(torch.float32), dim=1)[0]
                sampled_pos_inds_list, sampled_neg_inds_list = self.fg_bg_sampler(
                    [matched_idxs_per_image + 1], max_cls_logits_per_image
                )
            elif isinstance(self.fg_bg_sampler, BalancedPositiveNegativeSampler):
                sampled_pos_inds_list, sampled_neg_inds_list = self.fg_bg_sampler([matched_idxs_per_image + 1])
            else:
                raise NotImplementedError(
                    "Currently support torchvision BalancedPositiveNegativeSampler and monai HardNegativeSampler matcher. "
                    "Other types of sampler not supported. "
                    "Please override self.get_cls_train_sample_per_image(*) for your own sampler."
                )

            sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds_list, dim=0))[0]
            sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds_list, dim=0))[0]
            valid_idxs_per_image = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        return cls_logits_per_image[valid_idxs_per_image, :], gt_classes_target[valid_idxs_per_image, :]

    def get_box_train_sample_per_image(
        self,
        box_regression_per_image: Tensor,
        targets_per_image: Dict[str, Tensor],
        anchors_per_image: Tensor,
        matched_idxs_per_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get samples from one image for box regression losses computation.

        Args:
            box_regression_per_image: box regression result for one image, (sum(HWA), 2*self.spatial_dims)
            targets_per_image: a dict with at least two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image.
            anchors_per_image: anchors of one image,
                sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
                A = self.num_anchors_per_loc.
            matched_idxs_per_image: matched index, sized (sum(HWA),) or  (sum(HWDA),)

        Return:
            paired predicted and GT samples from one image for box regression losses computation
        """

        if torch.isnan(box_regression_per_image).any() or torch.isinf(box_regression_per_image).any():
            raise ValueError("NaN or Inf in predicted box regression.")

        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        num_gt_box = targets_per_image[self.target_box_key].shape[0]

        # if no GT box, return empty arrays
        if num_gt_box == 0:
            return box_regression_per_image[0:0, :], box_regression_per_image[0:0, :]

        # select only the foreground boxes
        # matched GT boxes for foreground anchors
        matched_gt_boxes_per_image = targets_per_image[self.target_box_key][
            matched_idxs_per_image[foreground_idxs_per_image]
        ].to(box_regression_per_image.device)
        # predicted box regression for foreground anchors
        box_regression_per_image = box_regression_per_image[foreground_idxs_per_image, :]
        # foreground anchors
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

        # encode GT boxes or decode predicted box regression before computing losses
        matched_gt_boxes_per_image_ = matched_gt_boxes_per_image
        box_regression_per_image_ = box_regression_per_image
        if self.encode_gt:
            matched_gt_boxes_per_image_ = self.box_coder.encode_single(matched_gt_boxes_per_image_, anchors_per_image)
        if self.decode_pred:
            box_regression_per_image_ = self.box_coder.decode_single(box_regression_per_image_, anchors_per_image)

        return box_regression_per_image_, matched_gt_boxes_per_image_


def retinanet_resnet50_fpn_detector(
    num_classes: int,
    anchor_generator: AnchorGenerator,
    returned_layers: Sequence[int] = (1, 2, 3),
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> RetinaNetDetector:
    """
    Returns a RetinaNet detector using a ResNet-50 as backbone, which can be pretrained
    from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`
    _.
    Args:
        num_classes: number of output classes of the model (excluding the background).
        anchor_generator: AnchorGenerator,
        returned_layers: returned layers to extract feature maps. Each returned layer should be in the range [1,4].
            len(returned_layers)+1 will be the number of extracted feature maps.
            There is an extra maxpooling layer LastLevelMaxPool() appended.
        pretrained: If True, returns a backbone pre-trained on 23 medical datasets
        progress: If True, displays a progress bar of the download to stderr

    Return:
        A RetinaNetDetector object with resnet50 as backbone

    Example:
        .. code-block:: python

            # define a naive network
            resnet_param = {
                "pretrained": False,
                "spatial_dims": 3,
                "n_input_channels": 2,
                "num_classes": 3,
                "conv1_t_size": 7,
                "conv1_t_stride": (2, 2, 2)
            }
            returned_layers = [1]
            anchor_generator = monai.apps.detection.utils.anchor_utils.AnchorGeneratorWithAnchorShape(
                feature_map_scales=(1, 2), base_anchor_shapes=((8,) * resnet_param["spatial_dims"])
            )
            detector = retinanet_resnet50_fpn_detector(
                **resnet_param, anchor_generator=anchor_generator, returned_layers=returned_layers
            )
    """

    backbone = resnet.resnet50(pretrained, progress, **kwargs)
    spatial_dims = len(backbone.conv1.stride)
    # number of output feature maps is len(returned_layers)+1
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=spatial_dims,
        pretrained_backbone=pretrained,
        trainable_backbone_layers=None,
        returned_layers=returned_layers,
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]
    size_divisible = [s * 2 * 2 ** max(returned_layers) for s in feature_extractor.body.conv1.stride]
    network = RetinaNet(
        spatial_dims=spatial_dims,
        num_classes=num_classes,
        num_anchors=num_anchors,
        feature_extractor=feature_extractor,
        size_divisible=size_divisible,
    )
    return RetinaNetDetector(network, anchor_generator)
