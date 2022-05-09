import math
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Callable, Union, Sequence

import torch
import torchvision
from torch import Tensor, nn
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler, Matcher

import monai
from monai.data import box_utils as box_ops
from monai.losses.focal_loss import FocalLoss
from monai.networks.layers.factories import Conv
from monai.networks.nets.resnet import resnet18,resnet34,resnet50
from monai.utils.module import look_up_option
from monai.utils import BlendMode, PytorchPadMode
from monai.inferers import SimpleInferer, SlidingWindowMultiOutputInferer

from . import _utils as det_utils
from .anchor_utils import AnchorGenerator
from .backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from .transform import GeneralizedRCNNTransform
from .sampler import HardNegativeSampler
from .ATSS_matcher import ATSSMatcher


# This script is modified from torchvision.models.detection.retinanet.py
__all__ = ["RetinaNet", "retinanet_resnet_fpn"]


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, spatial_dims: int = 3):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs, debug=False, fg_bg_sampler=None):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            "classification": self.classification_head.compute_loss(
                targets, head_outputs, matched_idxs, debug=debug, fg_bg_sampler=fg_bg_sampler
            ),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {"cls_logits": self.classification_head(x), "bbox_regression": self.regression_head(x)}


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01, spatial_dims: int = 3):
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        conv = []
        for _ in range(4):
            conv.append(conv_type(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(num_groups=8, num_channels=in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, conv_type):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = conv_type(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS
        self.cls_loss = None

    def compute_loss(self, targets, head_outputs, matched_idxs, debug=False, fg_bg_sampler=None):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs["cls_logits"]

        total_cls_logits = []
        total_gt_classes_target = []

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0


            num_foreground = foreground_idxs_per_image.sum()
            num_gt_box = targets_per_image["boxes"].numel()
            if debug:
                print(f"Number of positive anchors: {num_foreground}; Number of GT box: {num_gt_box}.")
            if num_gt_box > 0 and num_foreground<5:
                print(f"Only {num_foreground} positive anchors. Please decrease iou_thresh, adjust anchor settings, or change network first downsampling stride.")


            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image, # anchor idx
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]], # class label
            ] = 1.0

            if fg_bg_sampler is not None:
                if self.cls_loss == None:
                    # when there is balanced sampler, we use BCE loss (gamma=0)
                    self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
                # The input of fg_bg_sampler: list of tensors containing -1, 0 or positive values.
                # Each tensor corresponds to a specific image.
                # -1 values are ignored, 0 are considered as negatives and > 0 as positives.

                # matched_idxs_per_image (Tensor[int64]): an N tensor where N[i] is a matched gt in
                # [0, M - 1] or a negative value indicating that prediction i could not
                # be matched. BELOW_LOW_THRESHOLD = -1, BETWEEN_THRESHOLDS = -2
                if isinstance(fg_bg_sampler, HardNegativeSampler):
                    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler([matched_idxs_per_image + 1],cls_logits_per_image)
                else:
                    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler([matched_idxs_per_image + 1])
                sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
                sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
                # print(len(sampled_pos_inds), len(sampled_neg_inds))
                valid_idxs_per_image = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
            else:
                if self.cls_loss == None:
                    # when there is no balanced sampler, we use focal loss (gamma=2)
                    self.cls_loss = FocalLoss(reduction="mean", gamma=2.0)  # it has built-in sigmoid activation
                # find indices for which anchors should be ignored
                valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            total_cls_logits += cls_logits_per_image[valid_idxs_per_image,:]
            total_gt_classes_target += gt_classes_target[valid_idxs_per_image,:]

        total_cls_logits = torch.stack(total_cls_logits,dim=0)
        total_gt_classes_target = torch.stack(total_gt_classes_target,dim=0)
        # print(total_cls_logits[:5],total_gt_classes_target[:5])

        losses = self.cls_loss(
                    total_cls_logits,
                    total_gt_classes_target,
                ).to(total_cls_logits.dtype)
        return losses

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            all_cls_logits.append(cls_logits)

            if torch.isnan(cls_logits).any() or torch.isinf(cls_logits).any():
                raise ValueError("cls_logits is NaN or Inf.")

        return all_cls_logits


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors, spatial_dims: int = 3):
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        conv = []
        for _ in range(4):
            conv.append(conv_type(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.GroupNorm(num_groups=8, num_channels=in_channels))
            conv.append(nn.ReLU())

        self.conv = nn.Sequential(*conv)

        self.bbox_reg = conv_type(in_channels, num_anchors * 2 * spatial_dims, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, conv_type):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.box_coder = det_utils.BoxCoder(weights=(1.0,) * 2 * spatial_dims)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs, giou_loss=False):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs["bbox_regression"]

        total_bbox_regression = []
        total_target_regression = []

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            if targets_per_image["boxes"].shape[0] == 0:
                continue
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()
            if num_foreground < targets_per_image["boxes"].shape[0]:
                num_target = targets_per_image["boxes"].shape[0]
                print(f"Number of gt box is {num_target};\n Number of matched anchor is {num_foreground}.\n Please change anchor setting.")

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            # print(anchors_per_image[0],matched_gt_boxes_per_image[0])
            if torch.isnan(bbox_regression_per_image).any() or torch.isinf(bbox_regression_per_image).any():
                raise ValueError("bbox_regression_per_image is NaN or Inf.")

            if not giou_loss:
                # compute the regression targets for L1 loss
                target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
                if torch.isnan(target_regression).any() or torch.isinf(target_regression).any():
                    raise ValueError("target_regression is NaN or Inf.")

                total_bbox_regression.append( bbox_regression_per_image)
                total_target_regression.append( target_regression)
            else:
                # compute decoded box for GIoU Loss
                decode_bbox_regression_per_image = self.box_coder.decode_single(
                    bbox_regression_per_image, anchors_per_image
                )
                total_bbox_regression.append( decode_bbox_regression_per_image)
                total_target_regression.append(matched_gt_boxes_per_image)

        if len(total_bbox_regression) == 0:
            losses = 0.0
            return losses


        total_bbox_regression = torch.cat(total_bbox_regression,dim=0)
        total_target_regression = torch.cat(total_target_regression,dim=0)

        if not giou_loss:
            losses = torch.nn.functional.smooth_l1_loss(
                        total_bbox_regression, total_target_regression, beta=1. / 9, reduction="mean"
                    ).to(total_bbox_regression.dtype)
        else:
            losses = monai.losses.giou_loss.generalized_iou_loss(
                        total_bbox_regression, total_target_regression, reduction="mean"
                    ).to(total_bbox_regression.dtype)

        return losses

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            all_bbox_regression.append(bbox_regression)

            if torch.isnan(bbox_regression).any() or torch.isinf(bbox_regression).any():
                raise ValueError("bbox_regression is NaN or Inf.")

        return all_bbox_regression

class RetinaNet(nn.Module):
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        balanced_sampler_pos_fraction (float): positive fraction when do balanced sampling for classification head.
            If 0, will use Focal loss and not do balanced sampling
        hard_neg_sampler (bool): if True, will torchvision balanced sampler; if False, will use nnDection hard negative balanced sampler. default True
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import RetinaNet
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # RetinaNet needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((32, 64, 128, 256, 512),),
        >>>     aspect_ratios=((0.5, 1.0, 2.0),)
        >>> )
        >>>
        >>> # put the pieces together inside a RetinaNet model
        >>> model = RetinaNet(backbone,
        >>>                   num_classes=2,
        >>>                   anchor_generator=anchor_generator)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": Matcher,
        "fg_bg_sampler": BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        backbone,
        num_classes,
        spatial_dims: int,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        atss_matcher=True,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=1000,
        debug=False,
        balanced_sampler_pos_fraction=0,
        hard_neg_sampler=True,
        box_overlap_metric="iou",
        mirror_aggregation=False,
        **kwargs,
    ):
        super().__init__()

        self.spatial_dims = look_up_option(spatial_dims, supported=[2, 3])
        # If 2D, directly use torchvision functions
        if self.spatial_dims == 2:
            print("Call torchvision function torchvision.models.detection.RetinaNet for 2D network.")
            return torchvision.models.detection.RetinaNet(
                backbone,
                num_classes,
                # transform parameters
                min_size=kwargs.pop("min_size", 800),
                max_size=kwargs.pop("max_size", 1333),
                image_mean=kwargs.pop("image_mean", None),
                image_std=kwargs.pop("image_std", None),
                # Anchor parameters
                anchor_generator=anchor_generator,
                head=head,
                proposal_matcher=proposal_matcher,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                detections_per_img=detections_per_img,
                fg_iou_thresh=fg_iou_thresh,
                bg_iou_thresh=bg_iou_thresh,
                topk_candidates=topk_candidates,
            )

        # If 3D:
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.anchor_generator = anchor_generator

        if head is None:
            head = RetinaNetHead(backbone.out_channels, self.anchor_generator.num_anchors_per_location()[0], num_classes)
        self.head = head

        if proposal_matcher is None:
            if not atss_matcher:
                proposal_matcher = Matcher(
                    fg_iou_thresh,
                    bg_iou_thresh,
                    allow_low_quality_matches=True,
                )
            else:
                if box_overlap_metric == "giou":
                    proposal_matcher = ATSSMatcher(num_candidates=4, similarity_fn=box_ops.box_giou,center_in_gt=False,debug=debug)
                else:
                    proposal_matcher = ATSSMatcher(num_candidates=4, similarity_fn=box_ops.box_iou,center_in_gt=False,debug=debug)

        self.proposal_matcher = proposal_matcher

        if (balanced_sampler_pos_fraction != None) and balanced_sampler_pos_fraction>0 :
            if hard_neg_sampler:
                self.fg_bg_sampler = HardNegativeSampler(batch_size_per_image=64, positive_fraction=balanced_sampler_pos_fraction, pool_size=20, min_neg=16)
            else:
                self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image=64, positive_fraction=balanced_sampler_pos_fraction)
        else:
            self.fg_bg_sampler = None


        self.box_coder = det_utils.BoxCoder(weights=(1.0,) * 2 * spatial_dims)

        self.transform = GeneralizedRCNNTransform(size_divisible=[s*16 for s in self.backbone.body.conv1.stride])

        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.debug = debug
        self.mirror_aggregation = mirror_aggregation

        # used only on torchscript mode
        self._has_warned = False
        self.inferer = SimpleInferer()
        look_up_option(box_overlap_metric, ["iou", "giou"])
        self.box_overlap_metric = box_overlap_metric

    def define_sliding_window_inferer(
        self,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int = 1,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
    ):
        # If this function is not called, the default self.inferer = SimpleInferer()
        self.sliding_window_inferer = SlidingWindowMultiOutputInferer(
            roi_size, sw_batch_size, 0.5, mode, sigma_scale, padding_mode, cval, sw_device, device
        )

        return

    def switch_inferer(
        self, sliding_window=True
    ):
        if sliding_window:
            self.inferer = self.sliding_window_inferer
        else:
            self.inferer = SimpleInferer()
        return

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors,
                        num_anchors_per_level: Sequence[int] = None,
                        num_anchors_per_loc: int = None):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            if isinstance(self.proposal_matcher, Matcher):
                # if torhvision matcher
                if self.box_overlap_metric == "giou":
                    match_quality_matrix = box_ops.box_giou(targets_per_image["boxes"], anchors_per_image)
                else:
                    match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))
            else:
                # if nndetection ATSS matcher
                match_quality_matrix, matches = self.proposal_matcher(targets_per_image["boxes"],anchors_per_image,num_anchors_per_level,num_anchors_per_loc)
                matched_idxs.append(matches)

            if self.debug:
                print(
                    f"Max {self.box_overlap_metric} between anchors and gt boxes: {torch.max(match_quality_matrix)}")

            if torch.max(match_quality_matrix)==0:
                raise ValueError(f"No GT box overlaps with anchors. Please adjust anchor setting. GT boxes are {targets_per_image['boxes']}")

        return self.head.compute_loss(
            targets, head_outputs, anchors, matched_idxs, debug=self.debug, fg_bg_sampler=self.fg_bg_sampler
        )

    def post_cat_map(self, x, num_channel):
        # postprocessing for result map, used for both training and inference
        all_bbox_regression = []

        for bbox_regression in x:
            # 2D: Permute bbox regression output from (N, num_channel * A, H, W) to (N, HWA, num_channel).
            spatial_dims = self.spatial_dims
            if spatial_dims == 2:
                N, _, H, W = bbox_regression.shape
                bbox_regression = bbox_regression.view(N, -1, num_channel, H, W) #(N, A, num_channel, H, W)
                bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
                bbox_regression = bbox_regression.reshape(N, -1, num_channel)  # Size=(N, HWA, num_channel)
            # 3D: Permute bbox regression output from (N, num_channel * A, H, W, D) to (N, HWDA, num_channel).
            elif spatial_dims == 3:
                N, _, H, W, D = bbox_regression.shape
                bbox_regression = bbox_regression.view(N, -1, num_channel, H, W, D) #(N, A, num_channel, H, W, D)
                bbox_regression = bbox_regression.permute(0, 3, 4, 5, 1, 2)
                bbox_regression = bbox_regression.reshape(N, -1, num_channel)  # Size=(N, HWDA, num_channel)
            else:
                ValueError("Images can only be 2D or 3D.")

            if torch.isnan(bbox_regression).any() or torch.isinf(bbox_regression).any():
                raise ValueError("Concatenated result is NaN or Inf.")

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)

    def postprocess_detections(self, head_outputs, anchors, image_shapes, sigmoid=True):
        # postprocessing during inference
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]
        compute_dtype = class_logits[0].dtype

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                if sigmoid:
                    scores_per_level = torch.sigmoid(logits_per_level).flatten()
                else:
                    scores_per_level = logits_per_level.flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.to(torch.float32).topk(num_topk) # half precision not implemented for cpu float16
                topk_idxs = topk_idxs[idxs]

                # decode box
                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs].to(torch.float32), anchors_per_level[anchor_idxs]
                ) # half precision not implemented for cpu float16
                boxes_per_level, keep = box_ops.box_clip_to_image(boxes_per_level, image_shape, remove_empty = True)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level[keep])
                image_labels.append(labels_per_level[keep])

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            keep = []
            # non-maximum suppression
            # print('start nms')
            for c in range(self.num_classes):
                image_labels_c_idx = (image_labels == c).nonzero(as_tuple=False).flatten()
                keep_c = box_ops.non_max_suppression(
                    image_boxes[image_labels_c_idx, :], image_scores[image_labels_c_idx], self.nms_thresh, box_overlap_metric=self.box_overlap_metric
                )
                keep_c = image_labels_c_idx[keep_c[: self.detections_per_img]]
                keep.append(keep_c)
            keep = torch.cat(keep, dim=0)
            # print('end nms')

            detections.append(
                {
                    "boxes": image_boxes[keep].to(compute_dtype),
                    "scores": image_scores[keep].to(compute_dtype),
                    "labels": image_labels[keep],
                }
            )

        return detections

    def forward_network(self, images, sigmoid=True):
        features = self.backbone(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        # TODO: Do we want a list or a dict?
        features = list(features.values())
        head_outputs = self.head(features)
        head_outputs_list = head_outputs["cls_logits"]
        if sigmoid:
            head_outputs_list = [torch.sigmoid(h) for h in head_outputs_list]
        head_outputs_list += head_outputs["bbox_regression"]
        return head_outputs_list

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 6:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 6], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        # transform the input:
        # concat multiple images to a single batch, if size not same, will pad to the max size
        # image_sizes: a list that document the original size
        images, image_sizes, targets = self.transform(images, targets)

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                # check if any invalid target
                degenerate_boxes = boxes[:, self.spatial_dims:] <= boxes[:, :self.spatial_dims]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        # get the features from the backbone
        # compute the retinanet heads outputs using the features
        if self.training:
            head_outputs_list = self.forward_network(images,sigmoid=False)
        else:
            if not self.mirror_aggregation:
                head_outputs_list = self.inferer(images, self.forward_network)
                head_outputs_list = [h.to("cpu") for h in head_outputs_list]
            else:
                head_outputs_list = []
                num_flip = 0
                if self.spatial_dims == 2:
                    flip_axis_2_range = [0] # not flip z-axis
                else:
                    flip_axis_2_range = [0,1] # may flip z-axis
                for flip_axis_0 in [0,1]:
                    for flip_axis_1 in [0,1]:
                        for flip_axis_2 in flip_axis_2_range:
                            # denote whether to flip in each axis
                            if self.spatial_dims == 2:
                                flip_axis_all = [flip_axis_0,flip_axis_1]
                            else:
                                flip_axis_all = [flip_axis_0,flip_axis_1,flip_axis_2]
                            # flip image
                            flip_images = images.clone()
                            for axis in range(len(flip_axis_all)):
                                if flip_axis_all[axis] == 1:
                                    flip_images = flip_images.flip(dims=[2+axis])
                            head_outputs_list_flip = self.inferer(flip_images, self.forward_network)
                            # flip back result
                            for hh in range(len(head_outputs_list_flip)):
                                head_outputs_list_flip[hh] = head_outputs_list_flip[hh].to("cpu")
                                for axis in range(len(flip_axis_all)):
                                    if flip_axis_all[axis] == 1:
                                        head_outputs_list_flip[hh] = head_outputs_list_flip[hh].flip(dims=[2+axis])
                            # inverse the center shift of box regression
                            # after flipping, shift left becomes shift right
                            num_anchors = head_outputs_list_flip[-1].shape[1]//2//self.spatial_dims
                            for hh in range(len(head_outputs_list_flip) // 2,len(head_outputs_list_flip)):
                                for nn in range(num_anchors):
                                    for axis in range(len(flip_axis_all)):
                                        if flip_axis_all[axis] == 1:
                                            head_outputs_list_flip[hh][:,2*self.spatial_dims*nn+axis,...] = -head_outputs_list_flip[hh][:,2*self.spatial_dims*nn+axis,...]

                            # aggregate result
                            if len(head_outputs_list)==0:
                                head_outputs_list = head_outputs_list_flip
                            else:
                                for hh in range(len(head_outputs_list)):
                                    head_outputs_list[hh] = head_outputs_list[hh] + head_outputs_list_flip[hh]
                            num_flip += 1
                head_outputs_list = [h.to("cpu")/num_flip for h in head_outputs_list]
                del head_outputs_list_flip,flip_images

        anchor_dtype, anchor_device = head_outputs_list[0].dtype, head_outputs_list[0].device

        head_outputs = {}
        head_outputs["cls_logits"] = head_outputs_list[: len(head_outputs_list) // 2]
        head_outputs["bbox_regression"] = head_outputs_list[len(head_outputs_list) // 2 :]

        # create the set of anchors
        grid_sizes = [
            head_output_map.shape[-self.spatial_dims :]
            for head_output_map in head_outputs["cls_logits"]
        ] # anchor grid, 2D/3D

        anchors = self.anchor_generator(images.shape[-self.spatial_dims :], anchor_dtype, anchor_device, orig_image_size_list=image_sizes, grid_sizes=grid_sizes) # list, len(anchors) = batchsize

        num_anchors_per_level = [x.shape[2:].numel() for x in head_outputs["cls_logits"]]

        head_outputs["cls_logits"] = self.post_cat_map(head_outputs["cls_logits"], num_channel=self.num_classes)
        head_outputs["bbox_regression"] = self.post_cat_map(
            head_outputs["bbox_regression"], num_channel=2 * self.spatial_dims
        )

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None

            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors, num_anchors_per_level,self.anchor_generator.num_anchors_per_location()[0])
        else:
            # recover level sizes
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, image_sizes, sigmoid=False)
            detections = self.transform.postprocess(
                detections, [img.shape[-self.spatial_dims :] for img in images], image_sizes
            )

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)


model_urls = {
    "retinanet_resnet50_fpn_coco": "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
}

def retinanet_resnet_fpn(
    spatial_dims: int,
    backbone,
    pretrained=False,
    num_classes=2,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    **kwargs,
):
    """
    Constructs a RetinaNet model with a ResNet-50-FPN backbone.

    Reference: `"Focal Loss for Dense Object Detection" <https://arxiv.org/abs/1708.02002>`_.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Example::

        >>> model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    spatial_dims = look_up_option(spatial_dims, supported=[2, 3])

    # If 3D, we do not have pretrained detection model, only pretrained_backbone is available
    returned_layers = kwargs.pop("returned_layers", [1, 2, 3])

    pretrained = look_up_option(pretrained, supported=[False])

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    # skip P2 because it generates too many anchors (according to their paper)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=returned_layers, extra_blocks=None
    )

    model = RetinaNet(backbone, num_classes, spatial_dims=spatial_dims, **kwargs)
    return model
