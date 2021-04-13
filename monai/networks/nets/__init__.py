# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .ahnet import AHNet
from .autoencoder import AutoEncoder
from .basic_unet import BasicUNet, BasicUnet, Basicunet
from .classifier import Classifier, Critic, Discriminator
from .densenet import DenseNet, DenseNet121, DenseNet169, DenseNet201, DenseNet264
from .dynunet import DynUNet, DynUnet, Dynunet
from .efficientnet import EfficientNet, EfficientNetBN, drop_connect, get_efficientnet_image_size
from .fullyconnectednet import FullyConnectedNet, VarFullyConnectedNet
from .generator import Generator
from .highresnet import HighResBlock, HighResNet
from .regressor import Regressor
from .regunet import GlobalNet, LocalNet, RegUNet
from .segresnet import SegResNet, SegResNetVAE
from .senet import SENet, SENet154, SEResNet50, SEResNet101, SEResNet152, SEResNext50, SEResNext101
from .torchvision_fc import TorchVisionFullyConvModel
from .unet import UNet, Unet, unet
from .varautoencoder import VarAutoEncoder
from .vnet import VNet
