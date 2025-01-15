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

from .adversarial_loss import PatchAdversarialLoss
from .barlow_twins import BarlowTwinsLoss
from .cldice import SoftclDiceLoss, SoftDiceclDiceLoss
from .contrastive import ContrastiveLoss
from .deform import BendingEnergyLoss, DiffusionLoss
from .dice import (
    Dice,
    DiceCELoss,
    DiceFocalLoss,
    DiceLoss,
    GeneralizedDiceFocalLoss,
    GeneralizedDiceLoss,
    GeneralizedWassersteinDiceLoss,
    MaskedDiceLoss,
    dice_ce,
    dice_focal,
    generalized_dice,
    generalized_dice_focal,
    generalized_wasserstein_dice,
)
from .ds_loss import DeepSupervisionLoss
from .focal_loss import FocalLoss
from .giou_loss import BoxGIoULoss, giou
from .hausdorff_loss import HausdorffDTLoss, LogHausdorffDTLoss
from .image_dissimilarity import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss
from .multi_scale import MultiScaleLoss
from .nacl_loss import NACLLoss
from .perceptual import PerceptualLoss
from .spatial_mask import MaskedLoss
from .spectral_loss import JukeboxLoss
from .ssim_loss import SSIMLoss
from .sure_loss import SURELoss
from .tversky import TverskyLoss
from .unified_focal_loss import AsymmetricUnifiedFocalLoss
