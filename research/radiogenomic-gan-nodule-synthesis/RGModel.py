# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.parallel
from torch import nn

from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Conv, Norm, split_args

"""
An implementation of the Radiogenomic-GAN originally proposed by:

Z. Xu, X. Wang, H. Shin, D. Yang, H. Roth, F. Milletari, L. Zhang, D. Xu (2019) 
"Correlation via synthesis: end-to-end nodule image generation and radiogenomic map learning 
based on generative adversarial network. 2020. DOI: [1907.03728](https://arxiv.org/pdf/1907.03728.pdf)

Network architecture based on original MC GAN https://github.com/HYOJINPARK/MC_GAN
"""

# ############# AdaIN ############# #
# Adapted from https://github.com/naoto0804/pytorch-AdaIN


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    batch_size, channels = size[:2]
    feat = feat.contiguous()
    feat_var = feat.view(batch_size, channels, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


# ############# Generator ############# #


class SynthesisBlock(nn.Module):
    def __init__(
        self,
        dim,
        channels,
        norm=Norm.BATCH,
        act=(Act.LEAKYRELU, {"negative_slope": 0.02, "inplace": True}),
    ):
        super(SynthesisBlock, self).__init__()
        self.fg_block = nn.Sequential(
            Convolution(dim, channels, channels * 2, kernel_size=3, bias=False, act=act, norm=norm),
            Convolution(dim, channels * 2, channels * 2, kernel_size=3, bias=False, act=None, norm=norm),
        )
        self.channels = channels

    def forward(self, fg, bg, code):
        whole = self.fg_block(fg)
        out_switch_bg = (torch.tanh(whole[:, self.channels :]) + 1) / 2
        out_switch_fg = 1 - out_switch_bg

        fg_out = whole[:, : self.channels]
        bg_res = torch.mul(bg, out_switch_bg)
        fg_res = torch.mul(fg_out, out_switch_fg)
        fg_res = adain(fg_res, code)

        out_block = bg_res + fg_res
        return out_block, out_switch_bg


class SynthesisBlockDirect(nn.Module):
    def __init__(self, channels):
        super(SynthesisBlockDirect, self).__init__()
        self.channels = channels

    def forward(self, fg, bg, code):
        fg_out = fg[:, : self.channels]
        out_switch_fg = (torch.tanh(fg[:, self.channels :]) + 1) / 2
        out_switch_bg = 1 - out_switch_fg
        fg_code = fg_out
        bg_res = torch.mul(bg, out_switch_bg)
        fg_res = torch.mul(fg_code, out_switch_fg)
        out_block = bg_res + fg_res

        return out_block


class EncoderNet(nn.Module):
    def __init__(
        self,
        input_dim=5172,
        embed_dim=128,
        expansion_factor=4,
        act=(Act.LEAKYRELU, {"negative_slope": 0.2, "inplace": True}),
    ):
        super(EncoderNet, self).__init__()
        act, act_args = split_args(act)
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim * expansion_factor, bias=False),
            Act[act](**act_args),
            nn.Linear(embed_dim * expansion_factor, embed_dim, bias=False),
        )

    def forward(self, description):
        embedding = self.net(description)
        return embedding


class GenSynthStage(nn.Module):
    def __init__(self, dim, img_features, input_size, kernel_size, act=(Act.GLU, {"dim": 1}), norm=Norm.BATCH):
        super(GenSynthStage, self).__init__()
        self.dim = dim
        self.act = act
        self.norm = norm
        self.k_size = k = kernel_size
        self.img_features = nf = img_features

        self.fc4 = self._get_fc_layer(input_size, nf * k * k * 2)
        self.fc3 = self._get_fc_layer(input_size, nf * k * k)
        self.fc2 = self._get_fc_layer(input_size, nf * k * k // 2)
        self.fc1 = self._get_fc_layer(input_size, nf * k * k // 4)
        self.fc0 = self._get_fc_layer(input_size, nf * k * k // 8)

        self.img_block0 = Convolution(self.dim, 1, nf // 16, kernel_size=3, strides=1, norm=Norm.BATCH, act=None)
        self.img_block1 = Convolution(self.dim, nf // 16, nf // 8, kernel_size=3, strides=2, norm=Norm.BATCH, act=None)
        self.img_block2 = Convolution(self.dim, nf // 8, nf // 4, kernel_size=3, strides=2, norm=Norm.BATCH, act=None)
        self.img_block3 = Convolution(self.dim, nf // 4, nf // 2, kernel_size=3, strides=2, norm=Norm.BATCH, act=None)
        self.img_block4 = Convolution(self.dim, nf // 2, nf, kernel_size=3, strides=2, norm=Norm.BATCH, act=None)

        self.synthesis1 = SynthesisBlock(self.dim, nf)
        self.upsample1 = self._get_upsample_block(nf, nf // 2)

        self.synthesis2 = SynthesisBlock(self.dim, nf // 2)
        self.upsample2 = self._get_upsample_block(nf // 2, nf // 4)

        self.synthesis3 = SynthesisBlock(self.dim, nf // 4)
        self.upsample3 = self._get_upsample_block(nf // 4, nf // 8)

        self.synthesis4 = SynthesisBlock(self.dim, nf // 8)
        self.upsample4 = self._get_upsample_block(nf // 8, nf // 16)

    def _get_fc_layer(self, in_features, out_features, bias=False):
        act, act_args = split_args(self.act)
        fc = nn.Sequential(
            nn.Linear(in_features, out_features, bias), Norm["BATCH", 1](out_features), Act[act](**act_args)
        )
        return fc

    def _get_upsample_block(self, in_planes, out_planes):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Convolution(
                self.dim, in_planes, out_planes * 2, kernel_size=3, strides=1, bias=False, norm=self.norm, act=self.act
            ),
        )
        return block

    def forward(self, z_code, c_code, base_img):
        in_code = torch.cat((c_code, z_code), 1)

        fg_code4 = self.fc4(in_code)
        fg_code4 = fg_code4.view(-1, self.img_features, self.k_size, self.k_size)
        fg_code3 = self.fc3(in_code)
        fg_code3 = fg_code3.view(-1, self.img_features // 2, self.k_size, self.k_size)
        fg_code2 = self.fc2(in_code)
        fg_code2 = fg_code2.view(-1, self.img_features // 4, self.k_size, self.k_size)
        fg_code1 = self.fc1(in_code)
        fg_code1 = fg_code1.view(-1, self.img_features // 8, self.k_size, self.k_size)
        fg_code0 = self.fc0(in_code)
        fg_code0 = fg_code0.view(-1, self.img_features // 16, self.k_size, self.k_size)

        bg_code0 = self.img_block0(base_img)
        bg_code1 = self.img_block1(bg_code0)
        bg_code2 = self.img_block2(bg_code1)
        bg_code3 = self.img_block3(bg_code2)
        bg_code4 = self.img_block4(bg_code3)

        out_code1, _ = self.synthesis1(bg_code4, fg_code4, fg_code4)
        out_code1 = self.upsample1(out_code1)

        out_code2, _ = self.synthesis2(out_code1, bg_code3, fg_code3)
        out_code2 = self.upsample2(out_code2)

        out_code3, _ = self.synthesis3(out_code2, bg_code2, fg_code2)
        out_code3 = self.upsample3(out_code3)

        out_code4, out_switch_fg = self.synthesis4(out_code3, bg_code1, fg_code1)
        out_code4 = self.upsample4(out_code4)

        return out_code4, torch.mean(out_switch_fg, 1).unsqueeze(1)


class GenImgStage(nn.Module):
    def __init__(self, dim, img_features, input_size, kernel_size, output_channels=2):
        super(GenImgStage, self).__init__()
        self.k_size = kernel_size
        self.img = nn.Sequential(
            Conv["CONV", 2](img_features, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Act["TANH"](),
        )
        self.synth_out = SynthesisBlockDirect(1)
        self.fc = nn.Sequential(
            nn.Linear(input_size, self.k_size * self.k_size * 2, bias=False),
            Norm["BATCH", 1](self.k_size * self.k_size * 2),
            Act["GLU"](1),
        )

    def forward(self, h_code, z_code, c_code, base_img):
        in_code = torch.cat((c_code, z_code), 1)
        fg_code = self.fc(in_code)
        fg_code = fg_code.view(-1, 1, self.k_size, self.k_size)
        img_set = self.img(h_code)
        out_img = self.synth_out(img_set, base_img, fg_code)
        out_seg = img_set[:, 1, :]
        out_seg = torch.unsqueeze(out_seg, 1)

        return out_img, out_seg


class GenNet(nn.Module):
    def __init__(self, spatial_dim=2, img_features=32, embed_dim=128, latent_size=10, code_k_size=8):
        super(GenNet, self).__init__()
        input_size = embed_dim + latent_size
        self.ec_net = EncoderNet()
        self.synth_net = GenSynthStage(spatial_dim, img_features * 16, input_size, code_k_size)
        self.img_net = GenImgStage(spatial_dim, img_features, input_size, code_k_size)

    def forward(self, z_code, description, base_img):
        embedding = self.ec_net(description)
        img_code, fg_switch = self.synth_net(z_code, embedding, base_img)
        fake_img, fake_seg = self.img_net(img_code, z_code, embedding, base_img)

        return embedding, fake_img, fake_seg, fg_switch


# ############# Discriminator ############# #


class DiscNet(nn.Module):
    def __init__(
        self,
        spatial_dim=2,
        img_features=64,
        embed_dim=128,
        code_k_size=8,
        norm=Norm.BATCH,
        act=(Act.LEAKYRELU, {"negative_slope": 0.2, "inplace": True}),
    ):
        super(DiscNet, self).__init__()
        self.dim = spatial_dim
        self.img_features = img_features
        self.embed_dim = embed_dim
        self.k_size = code_k_size
        self.norm = norm
        self.act = act

        self.encoding_I = self._build_encode_net(self.img_features, 1)
        self.encoding_IS = self._build_encode_net(self.img_features, 2)
        self.encoding_ISC = self._get_img_encode_block(
            self.img_features * 8 + self.embed_dim, self.img_features * 8, padding=1, stride=1
        )

        self.logits_I = Conv["CONV", self.dim](self.img_features * 8, 1, kernel_size=self.k_size, stride=1)
        self.logits_IS = Conv["CONV", self.dim](self.img_features * 8, 1, kernel_size=self.k_size, stride=1)
        self.logits_ISC = Conv["CONV", self.dim](self.img_features * 8, 1, kernel_size=self.k_size, stride=1)

    def _get_img_encode_block(
        self, in_channel, out_channel, k_size=3, stride=2, padding=1, no_norm=False, no_act=False, bias=False
    ):
        block = nn.Sequential()
        block.add_module("conv", Conv["conv", self.dim](in_channel, out_channel, k_size, stride, padding, bias=bias))

        if not no_norm:
            norm, norm_args = split_args(self.norm)
            block.add_module("norm", Norm[norm, self.dim](out_channel, **norm_args))

        if not no_act:
            act, act_args = split_args(self.act)
            block.add_module("act", Act[act](**act_args))

        return block

    def _build_encode_net(self, ndf, input_dims):
        encode_img = nn.Sequential(
            self._get_img_encode_block(input_dims, ndf, no_norm=True),
            self._get_img_encode_block(ndf, ndf * 2),
            self._get_img_encode_block(ndf * 2, ndf * 4),
            self._get_img_encode_block(ndf * 4, ndf * 8),
        )
        return encode_img

    def forward(self, img, embedding, seg):
        # prepare input data tensors
        imgseg_pair = torch.cat((img, seg), 1)
        input_embed = embedding.view(-1, self.embed_dim, 1, 1)
        input_embed = input_embed.repeat(1, 1, self.k_size, self.k_size)

        # img realism
        encoding_i = self.encoding_I(img)
        # img seg realism
        encoding_is = self.encoding_IS(imgseg_pair)
        # img seg code realism
        input_isc = torch.cat((input_embed, encoding_is), 1)
        encoding_isc = self.encoding_ISC(input_isc)

        # calculate logits
        logits_i = self.logits_I(encoding_i)
        logits_is = self.logits_IS(encoding_is)
        logits_isc = self.logits_ISC(encoding_isc)

        return [logits_isc.view(-1), logits_i.view(-1), logits_is.view(-1)]
