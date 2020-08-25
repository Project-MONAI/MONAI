import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Act, Conv, Norm, split_args

CFG_TREE_BASE_SIZE = 128
CFG_GAN_Z_DIM = 10
CFG_GAN_EMBEDDING_DIM = 128
CFG_GAN_GF_DIM = 32
CFG_GAN_DF_DIM = 64
CFG_TEXT_DIMENSION = 5172


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat = feat.contiguous()
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


adain = adaptive_instance_normalization


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels dont divide 2!"
        nc = nc // 2
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


# ############## G networks ################################################

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU(),
    )
    return block


class Synthesis_Block(nn.Module):
    def __init__(self, channel_num):
        # print('DEBUG: Channel_Num %s' % channel_num)
        super(Synthesis_Block, self).__init__()
        self.fg_block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            nn.LeakyReLU(0.02, inplace=True),
            conv3x3(channel_num * 2, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
        )
        self.channel = channel_num

    def forward(self, fg, bg, code):
        # print('DEBUG: SBF Input fg %s, bg %s, code %s' % (bg.shape, bg.shape, code.shape))
        whole = self.fg_block(fg)
        # print('DEBUG: SBF Whole %s' % str(whole.shape))

        # print('DEBUG: SBF self.channel %s' % self.channel)
        # out_switch_bg = F.sigmoid(whole[:, self.channel:])
        out_switch_bg = (F.tanh(whole[:, self.channel :]) + 1) / 2
        # print('DEBUG: SBF out_switch_bg %s' % str(out_switch_bg.shape))
        out_switch_fg = 1 - out_switch_bg

        fg_out = whole[:, : self.channel]
        # print('DEBUG: SBF fg_out %s, out_switch_fg %s' % (fg_out.shape, out_switch_fg.shape))
        bg_res = torch.mul(bg, out_switch_bg)
        fg_res = torch.mul(fg_out, out_switch_fg)
        fg_res = adain(fg_res, code)

        out_block = bg_res + fg_res
        # print('DEBUG: END SYNTH BLOCK')
        return out_block, out_switch_bg


class Synthesis_Block_Direct(nn.Module):
    def __init__(self, channel_num):
        super(Synthesis_Block_Direct, self).__init__()
        self.channel = channel_num

    def forward(self, fg, bg, code):

        fg_out = fg[:, : self.channel]

        # out_switch_bg = F.sigmoid(fg[:, self.channel:])
        out_switch_fg = (F.tanh(fg[:, self.channel :]) + 1) / 2
        out_switch_bg = 1 - out_switch_fg

        # fg_code = adain(fg_out, code)
        fg_code = fg_out

        bg_res = torch.mul(bg, out_switch_bg)

        fg_res = torch.mul(fg_code, out_switch_fg)
        # fg_res = fg_code

        out_block = bg_res + fg_res

        return out_block


class EC_NET(nn.Module):
    def __init__(
        self, code_dim=5172, ef_dim=128,
    ):
        super(EC_NET, self).__init__()
        self.t_dim = code_dim
        self.ef_dim = ef_dim

        self.fc = nn.Sequential(
            nn.Linear(self.t_dim, self.ef_dim * 4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.ef_dim * 4, self.ef_dim, bias=False),
        )

        # print('Encoding from ', self.t_dim, 'to', self.ef_dim)

    def forward(self, text_embedding):
        c_code = self.fc(text_embedding)
        return c_code


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = CFG_GAN_Z_DIM + CFG_GAN_EMBEDDING_DIM

        self.k_size = CFG_TREE_BASE_SIZE // (2 ** 4)
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        k_size = self.k_size

        self.fc4 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size * 2, bias=False), nn.BatchNorm1d(ngf * k_size * k_size * 2), GLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size, bias=False), nn.BatchNorm1d(ngf * k_size * k_size), GLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size // 2, bias=False), nn.BatchNorm1d(ngf * k_size * k_size // 2), GLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size // 4, bias=False), nn.BatchNorm1d(ngf * k_size * k_size // 4), GLU()
        )

        self.fc0 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size // 8, bias=False), nn.BatchNorm1d(ngf * k_size * k_size // 8), GLU()
        )

        self.img_block0 = nn.Sequential(
            nn.Conv2d(1, ngf // 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ngf // 16)
        )

        self.img_block1 = nn.Sequential(
            nn.Conv2d(ngf // 16, ngf // 8, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ngf // 8)
        )
        self.img_block2 = nn.Sequential(
            nn.Conv2d(ngf // 8, ngf // 4, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ngf // 4)
        )
        self.img_block3 = nn.Sequential(
            nn.Conv2d(ngf // 4, ngf // 2, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ngf // 2)
        )
        self.img_block4 = nn.Sequential(
            nn.Conv2d(ngf // 2, ngf, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ngf)
        )
        self.synthesis1 = Synthesis_Block(ngf)
        self.upsample1 = upBlock(ngf, ngf // 2)

        self.synthesis2 = Synthesis_Block(ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)

        self.synthesis3 = Synthesis_Block(ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)

        self.synthesis4 = Synthesis_Block(ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

        self.synthesis5 = Synthesis_Block(ngf // 16)

    def forward(self, z_code, c_code, base_img):
        # print('DEBUG: Network Forward C_ shape %s, Z Shape %s' % (c_code.shape, z_code.shape))
        # DEBUG: Network Forward C_ shape torch.Size([112, 128]), Z Shape torch.Size([16, 10]) from save
        # DEBUG: Network Forward C_ shape torch.Size([16, 128]), Z Shape torch.Size([16, 10]) from forward
        in_code = torch.cat((c_code, z_code), 1)

        fg_code4 = self.fc4(in_code)
        fg_code4 = fg_code4.view(-1, self.gf_dim, self.k_size, self.k_size)
        fg_code3 = self.fc3(in_code)
        fg_code3 = fg_code3.view(-1, self.gf_dim // 2, self.k_size, self.k_size)
        fg_code2 = self.fc2(in_code)
        fg_code2 = fg_code2.view(-1, self.gf_dim // 4, self.k_size, self.k_size)
        fg_code1 = self.fc1(in_code)
        fg_code1 = fg_code1.view(-1, self.gf_dim // 8, self.k_size, self.k_size)
        fg_code0 = self.fc0(in_code)
        fg_code0 = fg_code0.view(-1, self.gf_dim // 16, self.k_size, self.k_size)

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

        # out_code5, out_switch_fg = self.synthesis5(out_code4, bg_code0, fg_code0)
        # return out_code5, torch.mean(out_switch_fg, 1).unsqueeze(1)
        return out_code4, torch.mean(out_switch_fg, 1).unsqueeze(1)


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf, out_dim):
        super(GET_IMAGE_G, self).__init__()
        self.in_dim = CFG_GAN_Z_DIM + CFG_GAN_EMBEDDING_DIM
        self.k_size = CFG_TREE_BASE_SIZE // (2 ** 4)
        self.gf_dim = ngf
        self.out_dim = out_dim
        self.img = nn.Sequential(conv3x3(ngf, out_dim), nn.Tanh())
        self.synthesisOut = Synthesis_Block_Direct(1)
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.k_size * self.k_size * 2, bias=False),
            nn.BatchNorm1d(self.k_size * self.k_size * 2),
            GLU(),
        )

    def forward(self, h_code, z_code, c_code, base_img):
        img_set = self.img(h_code)

        in_code = torch.cat((c_code, z_code), 1)
        fg_code = self.fc(in_code)
        fg_code = fg_code.view(-1, 1, self.k_size, self.k_size)

        out_img = self.synthesisOut(img_set, base_img, fg_code)

        out_seg = img_set[:, 1, :]
        out_seg = torch.unsqueeze(out_seg, 1)

        return out_img, out_seg

        # if self.out_dim > 1:
        #     out_img = img_set[:, :1, :]
        #     out_seg = img_set[:, 1, :]
        #     out_seg = torch.unsqueeze(out_seg, 1)
        #     return out_img, out_seg
        # else:
        #     return img_set


class GenNet(nn.Module):
    def __init__(self, gf_dim=32, ef_dim=128):
        super(GenNet, self).__init__()
        self.gf_dim = gf_dim  # CFG_GAN_GF_DIM  # 32
        self.ef_dim = ef_dim  # CFG_GAN_EMBEDDING_DIM  # 128

        self.ec_net = EC_NET()
        self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
        self.img_net1 = GET_IMAGE_G(self.gf_dim, 2)

    def forward(self, z_code, text_embedding, base_img):
        c_code = self.ec_net(text_embedding)
        h_code1, fg_switch1 = self.h_net1(z_code, c_code, base_img)
        fake_img1, fake_seg1 = self.img_net1(h_code1, z_code, c_code, base_img)

        return c_code, fake_img1, fake_seg1, fg_switch1


# ############## D networks ################################################
class DiscNet(nn.Module):
    def __init__(
        self,
        spatial_dim=2,
        df_dim=64,
        embed_dim=128,
        k_size=8,
        norm=Norm.BATCH,
        act=(Act.LEAKYRELU, {"negative_slope": 0.2, "inplace": True}),
    ):
        super(DiscNet, self).__init__()
        self.dim = spatial_dim
        self.df_dim = df_dim  # CFG_GAN_DF_DIM
        self.embed_dim = embed_dim  # CFG_GAN_EMBEDDING_DIM
        self.k_size = k_size  # CFG_TREE_BASE_SIZE // (2 ** 4) OR 128 // 16
        self.norm = norm
        self.act = act

        self.encoding_I = self._build_img_encode(self.df_dim, 1)
        self.encoding_IS = self._build_img_encode(self.df_dim, 2)
        self.encoding_ISC = self._get_encode_img_layer(self.df_dim * 8 + self.embed_dim, self.df_dim * 8, padding=1, stride=1)

        self.logits_I = Conv["CONV", self.dim](self.df_dim * 8, 1, kernel_size=self.k_size, stride=1)
        self.logits_IS = Conv["CONV", self.dim](self.df_dim * 8, 1, kernel_size=self.k_size, stride=1)
        self.logits_ISC = Conv["CONV", self.dim](self.df_dim * 8, 1, kernel_size=self.k_size, stride=1)

    def _get_encode_img_layer(
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

    def _build_img_encode(self, ndf, input_dims):
        encode_img = nn.Sequential(
            self._get_encode_img_layer(input_dims, ndf, no_norm=True),
            self._get_encode_img_layer(ndf, ndf * 2),
            self._get_encode_img_layer(ndf * 2, ndf * 4),
            self._get_encode_img_layer(ndf * 4, ndf * 8),
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
