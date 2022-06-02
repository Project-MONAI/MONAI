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
from monai.apps.detection.utils.predict_utils import ensure_dict_value_to_list_, predict_with_inferer
from monai.apps.detection.utils.anchor_utils import AnchorGenerator
from monai.apps.detection.utils.box_coder import BoxCoder
from monai.apps.detection.utils.box_selector import BoxSelector
from monai.apps.detection.utils.detector_utils import check_training_targets, preprocess_images
from monai.data.box_utils import box_iou
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
        # can be updated by self.set_sliding_window_inferer(*)
        self.inferer=None
        # can be updated by self.set_box_selector_parameters(*),
        self.box_selector = BoxSelector(
            box_overlap_metric=self.box_overlap_metric,
            score_thresh=0.05,
            topk_candidates_per_level=1000,
            nms_thresh=0.5,
            detections_per_img=300,
            apply_sigmoid=True,
        )
        # can be updated by self.set_sliding_window_inferer(*), self.set_custom_inferer(*), self.switch_to_sliding_window_inferer(*).
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
        Define sliding window inferer.
        """
        inferer = SlidingWindowInferer(
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
        self.inferer = inferer

    def switch_to_sliding_window_inferer(self, use_inferer: bool = False):
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
            if self.inferer is None:
                raise ValueError(
                    "`self.inferer` is not defined."
                    "Please refer to function self.set_sliding_window_inferer(*)."
                )
            else:
                self.use_inferer = True
        else:
            self.use_inferer = False


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
        # 1. check if input arguments are valid
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

        # 2. pad list of images to a single Tensor `images` with spatial size divisible by self.size_divisible.
        # image_sizes stores the original spatial_size of each image before padding.
        images, image_sizes = preprocess_images(input_images, self.spatial_dims, self.size_divisible)

        # 3. generate network outputs. Use inferer only in evaluation mode.
        if self.training or (not self.use_inferer):
            head_outputs = self.network(images)
            ensure_dict_value_to_list_(head_outputs)
        else:
            head_outputs = predict_with_inferer(images, self.network, keys=[self.cls_key, self.box_reg_key], inferer=self.inferer)

        # 4. Generate anchors. TO DO: cache anchors in the next version, as it usually remains the same during training.
        anchors = self.anchor_generator(images, head_outputs[self.cls_key])  # list, len(anchors) = batchsize
        # num_anchor_locs_per_level: list of HW or HWD for each level
        num_anchor_locs_per_level = [x.shape[2:].numel() for x in head_outputs[self.cls_key]]

        # 5. Reshape and concatenate head_outputs values from List[Tensor] to Tensor
        for key in [self.cls_key, self.box_reg_key]:
            # reshape to Tensor sized(B, sum(HWA), self.num_classes) or (B, sum(HWA), 2* self.spatial_dims)
            # A = self.num_anchors_per_loc
            head_outputs[key] = self.reshape_maps(head_outputs[key])

        # 6(1). if during training, return losses
        if self.training:
            losses = self.compute_loss(head_outputs, targets, anchors, num_anchor_locs_per_level)  # type: ignore
            return losses

        # 6(2). if during inference, return detection results
        detections = self.postprocess_detections(
            head_outputs, anchors, image_sizes, num_anchor_locs_per_level  # type: ignore
        )
        return detections

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
