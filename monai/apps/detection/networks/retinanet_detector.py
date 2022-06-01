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
from monai.apps.detection.utils.box_coder import BoxCoder
from monai.apps.detection.utils.detector_utils import check_training_targets, preprocess_images
from monai.data import box_utils
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import resnet
from monai.utils import BlendMode, PytorchPadMode, ensure_tuple_rep


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
              The network needs the input spatial_size to be divisible by size_divisible
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
        self,
        network,
        anchor_generator: AnchorGenerator,
        box_overlap_metric: Callable = box_utils.box_iou,
        debug: bool = False,
        **kwargs,
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

        self.box_overlap_metric = box_overlap_metric
        self.debug = debug

        # default setting for both training and inference
        # can be updated by self.set_box_coder_weights(*)
        self.box_coder = BoxCoder(weights=(1.0,) * 2 * self.spatial_dims)

        # default keys in the ground truth targets and predicted boxes,
        # can be updated by self.set_target_keys(*)
        self.target_box_key = "boxes"
        self.target_label_key = "labels"
        self.pred_score_key = self.target_label_key + "_scores"  # score key for the detected boxes

        # default setting for inference,
        # can be updated by self.set_inference_parameters(*),
        self.score_thresh = 0.5
        self.topk_candidates = 1000
        self.nms_thresh = 0.2
        self.detections_per_img = 300
        # can be updated by self.set_sliding_window_inferer(*), self.set_custom_inferer(*), self.switch_to_inferer(*).
        self.use_inferer = False

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
        Define sliding window inferer and set it as the inferer.
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
        self.use_inferer = True
        return

    def set_custom_inferer(self, inferer):
        """
        Set custom inferer.
        """
        self.inferer = inferer
        self.use_inferer = True

    def switch_to_inferer(self, use_inferer: bool = False):
        """
        Choose whether to use inferer.

        In most cases, we do not use inferer as it is more efficient to directly forward the network.
        But when images are large and cannot fit in the GPU, we need to use inferer such as sliding window inferer.


        Args:
            use_inferer: whether to use self.inferer.
                If False, will simply forward the network.
                If True, will use self.inferer, and requires
                ``self.set_sliding_window_inferer(*args)`` or ``self.set_custom_inferer(*args)`` to have been called before.
        """
        if use_inferer:
            if hasattr(self, "inferer"):
                self.use_inferer = True
            else:
                raise ValueError(
                    "`self.inferer` is not defined."
                    "Please refer to function self.set_sliding_window_inferer(*) or self.set_custom_inferer(*)."
                )
        else:
            self.use_inferer = False
        return

    def set_inference_parameters(self, score_thresh=0.05, topk_candidates=1000, nms_thresh=0.5, detections_per_img=300):
        """
        Using for inference. Set the parameters that are used for box selection during inference.
        The box selection is performed with the following steps:

            1) Discard boxes with scores less than self.score_thresh
            2) Keep boxes with top self.topk_candidates scores for each level
            3) Perform non-maximum suppression (NMS) on remaining boxes for each image, with overapping threshold nms_thresh.
            4) Keep boxes with top self.detections_per_img scores for each image

        Args:
            score_thresh: no box with scores less than score_thresh will be kept
            topk_candidates: max number of boxes to keep for each level
            nms_thresh: box overlapping threshold for NMS
            detections_per_img: max number of boxes to keep for each image
        """
        self.score_thresh = score_thresh
        self.topk_candidates = topk_candidates
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def forward(
        self, input_images: Union[List[Tensor], Tensor], targets: Union[List[Dict[str, Tensor]], None] = None
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Returns a dict of losses during training, or a list predicted dict of boxes and labels during inference.

        Args:
            input_images: The input to the model is expected to be a list of tensors, each of shape (C, H, W) or  (C, H, W, D),
                one for each image, and should be in 0-1 range. Different images can have different sizes.
                Or it can also be a Tensor sized (B, C, H, W) or  (B, C, H, W, D). In this case, all images have same size.
            targets: a list of dict. Each dict with two keys: self.target_box_key and self.target_label_key,
                ground-truth boxes present in the image (optional).

        Return:
            If training mode, will return a dict with at least two keys,
            including self.cls_key and self.box_reg_key, representing classification loss and box regression loss.

            If evaluation mode, will return a list of detection results.
            Each element corresponds to an images in ``input_images``, is a dict with at least three keys,
            including self.target_box_key, self.target_label_key, self.pred_score_key,
            representing predicted boxes, classification labels, and classification scores.

        """
        # check if input arguments are valid
        if self.training:
            check_training_targets(input_images, targets, self.spatial_dims, self.target_label_key, self.target_box_key)
            if not hasattr(self, "proposal_matcher"):
                raise AttributeError(
                    "Matcher is not set. Please refer to self.set_regular_matcher(*), "
                    "self.set_atss_matcher(*), or or self.set_custom_matcher(*)."
                )
            if self.fg_bg_sampler is None and self.debug:
                warnings.warn(
                    "No balanced sampler is used. Negative samples are likely to "
                    "be much more than positive samples. Please set balanced samplers with self.set_balanced_sampler(*) "
                    "and self.set_hard_negative_sampler(*), "
                    "or set classification loss function as Focal loss with self.set_cls_loss(*)"
                )

        # pad list of images to a single Tensor `images` with spatial size divisible by self.size_divisible.
        # image_sizes stores the original spatial_size of each image before padding.
        images, image_sizes = preprocess_images(input_images, self.spatial_dims, self.size_divisible)

        # generate network outputs. Use inferer only in evaluation mode.
        head_outputs = self.forward_network(images, use_inferer=((not self.training) and self.use_inferer))

        # post processing steps that both training and inference perform.
        head_outputs, anchors, num_anchor_locs_per_level = self.process_network_outputs_with_anchors(  # type: ignore
            images, head_outputs
        )

        # if during training, return losses
        if self.training:
            losses = self.compute_loss(head_outputs, targets, anchors, num_anchor_locs_per_level)  # type: ignore
            return losses

        # if during inference, return detection results
        detections = self.postprocess_detections(
            head_outputs, anchors, image_sizes, num_anchor_locs_per_level  # type: ignore
        )
        return detections

    def forward_network(self, images: Tensor, use_inferer: Union[bool, None] = None) -> Dict[str, List[Tensor]]:
        """
        Compute the output of network.

        Args:
            images: input of the network
            use_inferer: whether to use self.inferer. If not given, will assume False during training,
                and True during inference.

        Return:
            The output of the network.
                - It is a dictionary with at least two keys:
                  ``self.cls_key`` and ``self.box_reg_key``.
                - ``head_outputs[self.cls_key]`` should be List[Tensor]. Each Tensor represents
                  classification logits map at one resolution level,
                  sized (B, num_classes*A, H_i, W_i) or (B, num_classes*A, H_i, W_i, D_i),
                  A = self.num_anchors_per_loc.
                - ``head_outputs[self.box_reg_key]`` should be List[Tensor]. Each Tensor represents
                  box regression map at one resolution level,
                  sized (B, 2*spatial_dims*A, H_i, W_i)or (B, 2*spatial_dims*A, H_i, W_i, D_i).
                - ``len(head_outputs[self.cls_key]) == len(head_outputs[self.box_reg_key])``.
        """
        if use_inferer is None:
            use_inferer = not self.training

        # if not use_inferer, directly forward the network
        if not use_inferer:
            return self.ensure_network_outputs_values_list(self.network(images))

        # if use_inferer, we need to decompose the output dict into sequence,
        # then do infererence, finally reconstruct dict.
        head_outputs_sequence = self.inferer(images, self.network_sequence_output)
        head_outputs = {self.cls_key: list(head_outputs_sequence[: self.num_output_levels])}
        head_outputs[self.box_reg_key] = list(head_outputs_sequence[self.num_output_levels :])
        return head_outputs

    def ensure_network_outputs_values_list(
        self, head_outputs: Union[Dict[str, List[Tensor]], Dict[str, Tensor]]
    ) -> Dict[str, List[Tensor]]:
        """
        We expect the output of self.network to be Dict[str, List[Tensor]].
        If it is Dict[str, Tensor], this func converts it to Dict[str, List[Tensor]].

        Args:
            head_outputs: the outputs of self.network.
                - It is a dictionary with at least two keys:
                  ``self.cls_key`` and ``self.box_reg_key``.
                - ``head_outputs[self.cls_key]`` should be List[Tensor] or Tensor. Each Tensor represents
                  classification logits map at one resolution level,
                  sized (B, num_classes*A, H_i, W_i) or (B, num_classes*A, H_i, W_i, D_i),
                  A = self.num_anchors_per_loc.
                - ``head_outputs[self.box_reg_key]`` should be List[Tensor] or Tensor. Each Tensor represents
                  box regression map at one resolution level,
                  sized (B, 2*spatial_dims*A, H_i, W_i)or (B, 2*spatial_dims*A, H_i, W_i, D_i).
                - ``len(head_outputs[self.cls_key]) == len(head_outputs[self.box_reg_key])``.

        Return:
            a Dict[str, List[Tensor]]
        """
        if torch.jit.isinstance(head_outputs, Dict[str, List[Tensor]]):
            # if output of self.network should be a Dict[str, List[Tensor]], directly return it
            self.num_output_levels = len(head_outputs[self.cls_key])
            return head_outputs  # type: ignore

        if isinstance(head_outputs[self.cls_key], Tensor) and isinstance(head_outputs[self.box_reg_key], Tensor):
            # if output of self.network should be a Dict[str, Tensor], convert it to Dict[str, List[Tensor]]
            self.num_output_levels = 1
            head_outputs[self.cls_key] = [head_outputs[self.cls_key]]  # type: ignore
            head_outputs[self.box_reg_key] = [head_outputs[self.box_reg_key]]  # type: ignore
            return head_outputs  # type: ignore

        raise ValueError("The output of self.network should be Dict[str, List[Tensor]] or Dict[str, Tensor].")

    def network_sequence_output(self, images: Tensor) -> List[Tensor]:
        """
        Decompose the output of network (a dcit) into a sequence.

        Args:
            images: input of the network

        Return:
            network output list/tuple
        """
        head_outputs = self.ensure_network_outputs_values_list(self.network(images))

        if len(head_outputs[self.cls_key]) == len(head_outputs[self.box_reg_key]):
            return list(head_outputs[self.cls_key]) + list(head_outputs[self.box_reg_key])

        raise ValueError(f"Require len(head_outputs[{self.cls_key}]) == len(head_outputs[{self.box_reg_key}]).")

    def process_network_outputs_with_anchors(
        self, images: Tensor, head_outputs: Dict[str, List[Tensor]]
    ) -> Tuple[Dict[str, Tensor], List[Tensor], List[int]]:
        """
        Process network output for further processing, including generate anchors and reshape the head_outputs.
        This function is used in both training and inference.

        Args:
            images_shape: shape of network input images, (B, C_in, H, W) or (B, C_in, H, W, D)
            head_outputs: the outputs of self.network.
                - It is a dictionary with at least two keys:
                  ``self.cls_key`` and ``self.box_reg_key``.
                - ``head_outputs[self.cls_key]`` should be List[Tensor]. Each Tensor represents
                  classification logits map at one resolution level,
                  sized (B, num_classes*A, H_i, W_i) or (B, num_classes*A, H_i, W_i, D_i),
                  A = self.num_anchors_per_loc.
                - ``head_outputs[self.box_reg_key]`` should be List[Tensor]. Each Tensor represents
                  box regression map at one resolution level,
                  sized (B, 2*spatial_dims*A, H_i, W_i)or (B, 2*spatial_dims*A, H_i, W_i, D_i).
                - ``len(head_outputs[self.cls_key]) == len(head_outputs[self.box_reg_key])``.

        Return:
            - head_outputs_reshape, reshaped head_outputs. ``head_output_reshape[self.cls_key]`` is a Tensor
              sized (B, sum(HW(D)A), self.num_classes). ``head_output_reshape[self.box_reg_key]`` is a Tensor
              sized (B, sum(HW(D)A), 2*self.spatial_dims).
            - generated anchors, a list of Tensor. Each Tensor represents anchors for each image,
              sized (sum(HWA), 2*spatial_dims) or (sum(HWDA), 2*spatial_dims).
              A = self.num_anchors_per_loc.
            - number of anchor locations per output level, a list of HW or HWD for each level
        """
        anchors = self.anchor_generator(images, head_outputs[self.cls_key])  # list, len(anchors) = batchsize

        # x: result map tensor sized (B, C, H, W) or (B, C, H, W, D), different H, W, D for each level.
        # num_anchor_locs_per_level: list of HW or HWD for each level
        num_anchor_locs_per_level = [x.shape[2:].numel() for x in head_outputs[self.cls_key]]

        # reshaped results will have sized (B, sum(HWA)_across_levels, C/A)
        # A = self.num_anchors_per_loc
        head_outputs_reshape = {}
        for key in [self.cls_key, self.box_reg_key]
        head_outputs_reshape[key] = self.reshape_maps(
            head_outputs[key]
        )  # (B, sum(HWA), self.num_classes) or (B, sum(HWA), 2* self.spatial_dims)

        return head_outputs_reshape, anchors, num_anchor_locs_per_level

    def reshape_maps(self, result_maps: List[Tensor]) -> Tensor:
        """
        Concat network output map list to a single Tensor.
        This function is used in both training and inference.

        Args:
            result_maps: a list of Tensor, each Tensor is a (B, C, H, W) or (B, C, H, W, D) map.
                C = num_channel*self.num_anchors_per_loc

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

    def select_top_boxes_per_level(
        self,
        box_regression_per_level: Tensor,
        logits_per_level: Tensor,
        anchors_per_level: Tensor,
        image_shape: Sequence,
        need_sigmoid: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Select boxes with highest scores for one level at one image.

        Args:
            box_regression_per_level: Tensor sized (HWA, 2*spatial_dims) or (HWDA, 2*spatial_dims)
            logits_per_level: Tensor sized (HWA, self.num_classes) or (HWDA, self.num_classes)
            anchors_per_level: Tensor sized (HWA, 2*spatial_dims) or (HWDA, 2*spatial_dims)
            image_shape: spatial_size of this image

        Return:
            selected boxes, classification scores, labels
        """
        num_classes = logits_per_level.shape[-1]
        # apply sigmoid to classification logits if asked
        if need_sigmoid:
            scores_per_level = torch.sigmoid(logits_per_level.to(torch.float32)).flatten()
        else:
            scores_per_level = logits_per_level.flatten()

        # remove low scoring boxes
        keep_idxs = scores_per_level > self.score_thresh
        scores_per_level = scores_per_level[keep_idxs]
        topk_idxs = torch.where(keep_idxs)[0]

        # keep only topk scoring predictions
        num_topk = min(self.topk_candidates, topk_idxs.size(0))
        scores_per_level, idxs = scores_per_level.to(torch.float32).topk(
            num_topk
        )  # half precision not implemented for cpu float16
        topk_idxs = topk_idxs[idxs]

        # decode box
        anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
        labels_per_level = topk_idxs % num_classes

        boxes_per_level = self.box_coder.decode_single(
            box_regression_per_level[anchor_idxs].to(torch.float32), anchors_per_level[anchor_idxs]
        )  # half precision not implemented for cpu float16
        boxes_per_level, keep = box_utils.clip_boxes_to_image(  # type: ignore
            boxes_per_level, image_shape, remove_empty=True
        )

        return boxes_per_level, scores_per_level[keep], labels_per_level[keep]  # type: ignore

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

        The box selection is performed with the following steps:

            1) Discard boxes with scores less than self.score_thresh
            2) Keep boxes with top self.topk_candidates scores for each level
            3) Perform non-maximum suppression on remaining boxes for each image, with overapping threshold nms_thresh.
            4) Keep boxes with top self.detections_per_img scores for each image

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

        class_logits = split_head_outputs[self.cls_key]  # List[Tensor], each sized (HWA, self.num_classes)
        box_regression = split_head_outputs[self.box_reg_key]  # List[Tensor], each sized (HWA, 2*spatial_dims)
        compute_dtype = class_logits[0].dtype

        num_images = len(image_sizes)  # B

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = split_anchors[index], image_sizes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                boxes_per_level, scores_per_level, labels_per_level = self.select_top_boxes_per_level(
                    box_regression_per_level,
                    logits_per_level,
                    anchors_per_level,
                    image_shape,
                    need_sigmoid=need_sigmoid,
                )

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes_t = torch.cat(image_boxes, dim=0)
            image_scores_t = torch.cat(image_scores, dim=0)
            image_labels_t = torch.cat(image_labels, dim=0)

            # non-maximum suppression on detected boxes from all levels
            keep = []
            for c in range(self.num_classes):
                # NMS for boxes with label c
                image_labels_c_idx = (image_labels_t == c).nonzero(as_tuple=False).flatten()
                keep_c = box_utils.non_max_suppression(
                    image_boxes_t[image_labels_c_idx, :],
                    image_scores_t[image_labels_c_idx],
                    self.nms_thresh,
                    box_overlap_metric=self.box_overlap_metric,
                )
                keep_c = image_labels_c_idx[keep_c[: self.detections_per_img]]  # type: ignore
                keep.append(keep_c)
            keep_t = torch.cat(keep, dim=0)

            detections.append(
                {
                    self.target_box_key: image_boxes_t[keep_t].to(compute_dtype),
                    self.pred_score_key: image_scores_t[keep_t].to(compute_dtype),
                    self.target_label_key: image_labels_t[keep_t],
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
        return {}


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
