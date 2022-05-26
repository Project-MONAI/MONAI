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

from .acti_norm import ADN
from .activation import MemoryEfficientSwish, Mish, Swish
from .aspp import SimpleASPP
from .convolutions import Convolution, ResidualUnit
from .crf import CRF
from .dints_block import ActiConvNormBlock, FactorizedIncreaseBlock, FactorizedReduceBlock, P3DActiConvNormBlock
from .downsample import MaxAvgPool
from .dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock, get_output_padding, get_padding
from .fcn import FCN, GCN, MCFCN, Refine
from .localnet_block import LocalNetDownSampleBlock, LocalNetFeatureExtractorBlock, LocalNetUpSampleBlock
from .mlp import MLPBlock
from .patchembedding import PatchEmbed, PatchEmbeddingBlock
from .regunet_block import RegistrationDownSampleBlock, RegistrationExtractionBlock, RegistrationResidualConvBlock
from .segresnet_block import ResBlock
from .selfattention import SABlock
from .squeeze_and_excitation import (
    ChannelSELayer,
    ResidualSELayer,
    SEBlock,
    SEBottleneck,
    SEResNetBottleneck,
    SEResNeXtBottleneck,
)
from .transformerblock import TransformerBlock
from .unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from .upsample import SubpixelUpsample, Subpixelupsample, SubpixelUpSample, Upsample, UpSample
from .warp import DVF2DDF, Warp
