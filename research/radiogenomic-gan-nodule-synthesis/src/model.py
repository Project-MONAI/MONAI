
import torch
import torch.nn as nn
import torch.nn.parallel
# from src.miscc.config import cfg
CFG_TREE_BASE_SIZE = 128
CFG_GAN_Z_DIM = 10
CFG_GAN_EMBEDDING_DIM = 128
CFG_GAN_GF_DIM = 32
CFG_TREE_BRANCH_NUM = 1
CFG_GAN_DF_DIM = 64
CFG_TEXT_DIMENSION = 5172
from src.miscc.utils import adaptive_instance_normalization as adain
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class Synthesis_Block(nn.Module):
    def __init__(self, channel_num):
        # print('DEBUG: Channel_Num %s' % channel_num)
        super(Synthesis_Block, self).__init__()
        self.fg_block = nn.Sequential(
            conv3x3(channel_num, channel_num*2),
            nn.BatchNorm2d(channel_num*2),
            nn.LeakyReLU(0.02, inplace=True),
            conv3x3(channel_num*2, channel_num*2),
            nn.BatchNorm2d(channel_num*2)
        )
        self.channel = channel_num

    def forward(self, fg, bg, code):
        # print('DEBUG: SBF Input fg %s, bg %s, code %s' % (bg.shape, bg.shape, code.shape))
        whole = self.fg_block(fg)
        # print('DEBUG: SBF Whole %s' % str(whole.shape))

        # print('DEBUG: SBF self.channel %s' % self.channel)
        # out_switch_bg = F.sigmoid(whole[:, self.channel:])
        out_switch_bg = (F.tanh(whole[:, self.channel:])+1)/2
        # print('DEBUG: SBF out_switch_bg %s' % str(out_switch_bg.shape))
        out_switch_fg = 1-out_switch_bg

        fg_out = whole[:,:self.channel]
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

        fg_out = fg[:,:self.channel]

        #out_switch_bg = F.sigmoid(fg[:, self.channel:])
        out_switch_fg = (F.tanh(fg[:, self.channel:])+1)/2
        out_switch_bg = 1-out_switch_fg

        #fg_code = adain(fg_out, code)
        fg_code = fg_out

        bg_res = torch.mul(bg, out_switch_bg)

        fg_res = torch.mul(fg_code, out_switch_fg)
        #fg_res = fg_code

        out_block = bg_res + fg_res

        return out_block

class EC_NET(nn.Module):

    def __init__(self):
        super(EC_NET, self).__init__()
        self.t_dim = CFG_TEXT_DIMENSION
        self.ef_dim = CFG_GAN_EMBEDDING_DIM

        self.fc = nn.Sequential(
            nn.Linear(self.t_dim, self.ef_dim * 4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.ef_dim * 4, self.ef_dim, bias=False))

        # print('Encoding from ', self.t_dim, 'to', self.ef_dim)

    def forward(self, text_embedding):
        c_code = self.fc(text_embedding)
        return c_code


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = CFG_GAN_Z_DIM + CFG_GAN_EMBEDDING_DIM

        self.k_size = CFG_TREE_BASE_SIZE // (2**4)
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        k_size =self.k_size

        self.fc4 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size * 2, bias=False),
            nn.BatchNorm1d(ngf * k_size * k_size * 2),
            GLU())

        self.fc3 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size, bias=False),
            nn.BatchNorm1d(ngf * k_size * k_size),
            GLU())

        self.fc2 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size // 2, bias=False),
            nn.BatchNorm1d(ngf * k_size * k_size // 2),
            GLU())

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size // 4, bias=False),
            nn.BatchNorm1d(ngf * k_size * k_size // 4),
            GLU())

        self.fc0 = nn.Sequential(
            nn.Linear(in_dim, ngf * k_size * k_size // 8, bias=False),
            nn.BatchNorm1d(ngf * k_size * k_size // 8),
            GLU())

        self.img_block0 = nn.Sequential(
            nn.Conv2d(1, ngf // 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf // 16)
        )

        self.img_block1 = nn.Sequential(
            nn.Conv2d(ngf // 16, ngf // 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf // 8)
        )
        self.img_block2 = nn.Sequential(
            nn.Conv2d(ngf // 8, ngf // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf // 4)
        )
        self.img_block3 = nn.Sequential(
            nn.Conv2d(ngf // 4, ngf // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf // 2)
        )
        self.img_block4 = nn.Sequential(
            nn.Conv2d(ngf // 2, ngf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
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
        fg_code3 = fg_code3.view(-1, self.gf_dim//2, self.k_size, self.k_size)
        fg_code2 = self.fc2(in_code)
        fg_code2 = fg_code2.view(-1, self.gf_dim//4, self.k_size, self.k_size)
        fg_code1 = self.fc1(in_code)
        fg_code1 = fg_code1.view(-1, self.gf_dim//8, self.k_size, self.k_size)
        fg_code0 = self.fc0(in_code)
        fg_code0 = fg_code0.view(-1, self.gf_dim//16, self.k_size, self.k_size)

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
        self.img = nn.Sequential(
            conv3x3(ngf, out_dim),
            nn.Tanh())
        self.synthesisOut = Synthesis_Block_Direct(1)
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.k_size * self.k_size * 2, bias=False),
            nn.BatchNorm1d(self.k_size * self.k_size * 2),
            GLU())


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


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = CFG_GAN_GF_DIM
        self.define_module()
        self.ef_dim = CFG_GAN_EMBEDDING_DIM

    def define_module(self):

        self.ec_net = EC_NET()

        if CFG_TREE_BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.gf_dim, 2)


    def forward(self, z_code, text_embedding, base_img):

        c_code = self.ec_net(text_embedding)

        fake_imgs, fake_segs, fg_switchs = [], [], []
        # print('DEBUG: G_NET 311 Forward C %s, Z %s' % (c_code.shape, z_code.shape))
        h_code1, fg_switch1 = self.h_net1(z_code, c_code, base_img)
        fake_img1, fake_seg1 = self.img_net1(h_code1, z_code, c_code, base_img)

        fake_imgs.append(fake_img1)
        fake_segs.append(fake_seg1)
        fg_switchs.append(fg_switch1)

        return c_code, fake_imgs, fake_segs, fg_switchs


# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image(ndf, indim=1):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(indim, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# For 64 x 64 images
class D_NET(nn.Module):
    def __init__(self):
        super(D_NET, self).__init__()
        self.df_dim = CFG_GAN_DF_DIM
        self.ef_dim = CFG_GAN_EMBEDDING_DIM
        self.k_size = CFG_TREE_BASE_SIZE // (2**4)
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        k_size = self.k_size
        indim = 1
        self.img_code_s16 = encode_image(ndf)
        self.imgseg_code_s16 = encode_image(ndf, indim+1)

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)

        self.logits_IST = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=k_size, stride=1))
        self.logits_I = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=k_size, stride=1))
        self.logits_IS = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=k_size, stride=1))

    def forward(self, img, txt, seg):
        imgseg_pair = torch.cat((img, seg), 1)
        txt_code = txt.view(-1, self.ef_dim, 1, 1)
        txt_code = txt_code.repeat(1, 1, self.k_size, self.k_size)

        # only img
        img_code = self.img_code_s16(img)
        # img seg pair
        imgseg_code = self.imgseg_code_s16(imgseg_pair)

        # imgsegtxt
        h_c_code1 = torch.cat((txt_code, imgseg_code), 1)
        h_c_code1 = self.jointConv(h_c_code1)

        D_imgsegtxt = self.logits_IST(h_c_code1)
        D_img = self.logits_I(img_code)
        D_imgseg = self.logits_IS(imgseg_code)
        return [D_imgsegtxt.view(-1), D_img.view(-1), D_imgseg.view(-1)]

