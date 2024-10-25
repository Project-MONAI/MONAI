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

from .ahnet import AHnet, Ahnet, AHNet
from .attentionunet import AttentionUnet
from .autoencoder import AutoEncoder
from .autoencoderkl import AutoencoderKL
from .basic_unet import BasicUNet, BasicUnet, Basicunet, basicunet
from .basic_unetplusplus import BasicUNetPlusPlus, BasicUnetPlusPlus, BasicunetPlusPlus, basicunetplusplus
from .classifier import Classifier, Critic, Discriminator
from .controlnet import ControlNet
from .daf3d import DAF3D
from .densenet import (
    DenseNet,
    Densenet,
    DenseNet121,
    Densenet121,
    DenseNet169,
    Densenet169,
    DenseNet201,
    Densenet201,
    DenseNet264,
    Densenet264,
    densenet121,
    densenet169,
    densenet201,
    densenet264,
)
from .diffusion_model_unet import DiffusionModelUNet
from .dints import DiNTS, TopologyConstruction, TopologyInstance, TopologySearch
from .dynunet import DynUNet, DynUnet, Dynunet
from .efficientnet import (
    BlockArgs,
    EfficientNet,
    EfficientNetBN,
    EfficientNetBNFeatures,
    EfficientNetEncoder,
    drop_connect,
    get_efficientnet_image_size,
)
from .flexible_unet import FLEXUNET_BACKBONE, FlexibleUNet, FlexUNet, FlexUNetEncoderRegister
from .fullyconnectednet import FullyConnectedNet, VarFullyConnectedNet
from .generator import Generator
from .highresnet import HighResBlock, HighResNet
from .hovernet import Hovernet, HoVernet, HoVerNet, HoverNet
from .milmodel import MILModel
from .netadapter import NetAdapter
from .patchgan_discriminator import MultiScalePatchDiscriminator, PatchDiscriminator
from .quicknat import Quicknat
from .regressor import Regressor
from .regunet import GlobalNet, LocalNet, RegUNet
from .resnet import (
    ResNet,
    ResNetBlock,
    ResNetBottleneck,
    ResNetEncoder,
    ResNetFeatures,
    get_medicalnet_pretrained_resnet_args,
    get_pretrained_resnet_medicalnet,
    resnet10,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnet200,
)
from .segresnet import SegResNet, SegResNetVAE
from .segresnet_ds import SegResNetDS, SegResNetDS2
from .senet import (
    SENet,
    SEnet,
    Senet,
    SENet154,
    SEnet154,
    Senet154,
    SEResNet50,
    SEresnet50,
    Seresnet50,
    SEResNet101,
    SEresnet101,
    Seresnet101,
    SEResNet152,
    SEresnet152,
    Seresnet152,
    SEResNext50,
    SEResNeXt50,
    SEresnext50,
    Seresnext50,
    SEResNext101,
    SEResNeXt101,
    SEresnext101,
    Seresnext101,
    senet154,
    seresnet50,
    seresnet101,
    seresnet152,
    seresnext50,
    seresnext101,
)
from .spade_autoencoderkl import SPADEAutoencoderKL
from .spade_diffusion_model_unet import SPADEDiffusionModelUNet
from .spade_network import SPADENet
from .swin_unetr import PatchMerging, PatchMergingV2, SwinUNETR
from .torchvision_fc import TorchVisionFCModel
from .transchex import BertAttention, BertMixedLayer, BertOutput, BertPreTrainedModel, MultiModal, Pooler, Transchex
from .transformer import DecoderOnlyTransformer
from .unet import UNet, Unet
from .unetr import UNETR
from .varautoencoder import VarAutoEncoder
from .vista3d import VISTA3D, vista3d132
from .vit import ViT
from .vitautoenc import ViTAutoEnc
from .vnet import VNet
from .voxelmorph import VoxelMorph, VoxelMorphUNet
from .vqvae import VQVAE
