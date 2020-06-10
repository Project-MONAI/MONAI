from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision.utils import make_grid
import numpy as np
import os
import time
import cv2
from copy import deepcopy
import random
import nibabel as nib
import pickle

# from miscc.config import cfg
CFG_TREE_BRANCH_NUM = 1
CFG_TRAIN_NET_G = ''
CFG_TRAIN_NET_D = ''
CFG_TRAIN_DISCRIMINATOR_LR = 0.0001
CFG_TRAIN_GENERATOR_LR = 0.0001
CFG_TRAIN_FLAG = True
CFG_GPU_ID = '0'
CFG_TRAIN_MAX_EPOCH = 500
CFG_TRAIN_BATCH_SIZE = 16
CFG_CUDA = True
CFG_TRAIN_COEFF_ITS_LOSS = 10.0
CFG_TRAIN_COEFF_I_LOSS = 8.0
CFG_TRAIN_COEFF_IS_LOSS = 1.0
CFG_TRAIN_COEFF_RC_LOSS = 100.0
CFG_GAN_Z_DIM = 10
CFG_TRAIN_SNAPSHOT_INTERVAL = 500
from src.miscc.utils import mkdir_p

from src.model import G_NET, D_NET
import torchvision.transforms as transforms
from skimage.transform import resize

################################   NOTICE   ###########################################

# This code is based from stackGAN ++
# https://github.com/hanzhanggit/StackGAN-v2

#######################################################################################

# ################## Shared functions ###################
def compute_mean_covariance(img):

    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    img_hat_transpose = img_hat.transpose(1, 2)
    covariance = torch.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def negative_log_posterior_probability(predictions, num_splits=1):

    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network(gpus):
    netG = G_NET()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    # print(netG)
    print('instantiated Generator')

    netsD = []
    if CFG_TREE_BRANCH_NUM > 0:
        netsD.append(D_NET())

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
        print('instantiated Discriminator')
        # print(netsD[i])
    print('# of netsD', len(netsD))

    count = 0
    if CFG_TRAIN_NET_G != '':
        state_dict = torch.load(CFG_TRAIN_NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', CFG_TRAIN_NET_G)

        istart = CFG_TRAIN_NET_G.rfind('_') + 1
        iend = CFG_TRAIN_NET_G.rfind('.')
        count = CFG_TRAIN_NET_G[istart:iend]
        count = int(count) + 1

    if CFG_TRAIN_NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (CFG_TRAIN_NET_D, i))
            state_dict = torch.load('%s%d.pth' % (CFG_TRAIN_NET_D, i))
            netsD[i].load_state_dict(state_dict)

    return netG, netsD, len(netsD), count


def define_optimizers(netG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=CFG_TRAIN_DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    optimizerG = optim.Adam(netG.parameters(),
                            lr=CFG_TRAIN_GENERATOR_LR,
                            betas=(0.5, 0.999))
    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir):
    load_params(netG, avg_param_G)
    torch.save( netG.state_dict(),'%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)):
        netD = netsD[i]
        torch.save(netD.state_dict(),'%s/netD%d.pth' % (model_dir, i))
    print('Save G/Ds models.')


def largest_object(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = 0
    for i in range(1, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 1
    return(img2)


def load_test(data_dir):
    embedding_filename = '/infos/RNA_3D.pickle'
    with open(data_dir + embedding_filename, 'rb') as f:
        embeddings = pickle.load(f, encoding='latin1')
        embeddings = np.array(embeddings)
        embeddings = np.float32(embeddings)

    filepath = os.path.join(data_dir, 'infos/imagename_3D.pickle')
    with open(filepath, 'rb') as f:
        imagenames = pickle.load(f, encoding='latin1')

    # print('len imgnames: ', len(imagenames))
    # print('imgnames: ', imagenames)

    # Write out first 16 embedding images and their embeddings as reference
    image_test = []
    image_ref = []
    embedding_test = []
    image_ct = 0
    i = 0
    imagepath = os.path.join(data_dir, 'images', imagenames[i].decode("utf-8"), '5.nii.gz')
    IMAGE_MAX_COUNT = CFG_TRAIN_BATCH_SIZE
    while image_ct<IMAGE_MAX_COUNT:
        while (not os.path.isfile(imagepath)):
            i = i+1
            imagepath = os.path.join(data_dir, 'images', imagenames[i].decode("utf-8"), '5.nii.gz')

        img = nib.load(imagepath)
        affine = img.affine
        img = np.array(img.dataobj)
        img = img.astype(np.float64)
        re_img = resize(img, (128, 128))
        image_ref.append(re_img)

        if image_ct<IMAGE_MAX_COUNT:
            imagepath = os.path.join(data_dir, 'base_images', imagenames[i].decode("utf-8"), '5.nii.gz')
            img = nib.load(imagepath)
            img = np.array(img.dataobj)
            img = img.astype(np.float64)
            re_img = resize(img, (128, 128))
            image_test.append(re_img)

        embedding_test.append(embeddings[i])

        image_ct = image_ct+1
        i = i+1
        imagepath = os.path.join(data_dir, 'images', imagenames[i].decode("utf-8"), '5.nii.gz')


    # imagepath = os.path.join(data_dir, 'base_images', 'test/test.nii')
    # img = nib.load(imagepath)
    # img = np.array(img.dataobj)
    # img = img.astype(np.float64)
    # re_img = resize(img, (128, 128))
    # image_test.append(re_img)


    image_ref = torch.from_numpy(np.array(image_ref))
    image_test = torch.from_numpy(np.array(image_test))
    embedding_test = torch.from_numpy(np.array(embedding_test))
    embedding_all = torch.from_numpy(np.array(embeddings))

    image_ref = image_ref.type(torch.FloatTensor)
    image_ref = torch.clamp(image_ref, min=-1000, max=500)
    image_ref = torch.add(image_ref, 1000)
    image_ref = torch.unsqueeze(image_ref, 1)
    for i in range(IMAGE_MAX_COUNT):
        image_ref[i,:,:,:] = transforms.Normalize(mean=[750], std=[750])(image_ref[i,:,:,:])

    image_test = image_test.type(torch.FloatTensor)
    image_test = torch.clamp(image_test, min=-1000, max=500)
    image_test = torch.add(image_test, 1000)
    image_test = torch.unsqueeze(image_test, 1)
    for i in range(IMAGE_MAX_COUNT):
        image_test[i,:,:,:] = transforms.Normalize(mean=[750], std=[750])(image_test[i,:,:,:])

    grid = make_grid(image_ref)
    ndarr = grid.mul_(750).add_(-250).clamp_(-1000, 1000).permute(2, 1, 0).cpu().numpy()
    im_out = ndarr[:, :, 0]
    filepath = os.path.join(data_dir, 'Image_Ref.nii.gz')
    nib.save(nib.Nifti1Image(im_out, affine),filepath)

    print('DEBUG: COMPLETE LOAD TEST DATA IN TRAINER.PY')

    return image_test, embedding_test, embedding_all, image_ref, affine


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize, data_dir):
        if CFG_TRAIN_FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.embed_dir = os.path.join(output_dir, 'Embedding')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.embed_dir)

        self.img_test, self.embedding_test, self.embedding_all, self.img_ref, self.affine = load_test(data_dir)

        self.output_dir = output_dir
        s_gpus = CFG_GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = CFG_TRAIN_BATCH_SIZE * self.num_gpus
        self.max_epoch = CFG_TRAIN_MAX_EPOCH

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.SNAPSHOT_INTERVAL = CFG_TRAIN_SNAPSHOT_INTERVAL
        self.img_size = imsize // (2 ** (CFG_TREE_BRANCH_NUM - 1))

    def prepare_data(self, data):
        imgs, segs, w_imgs, w_segs, t_embedding, _, = data

        this_batch = imgs[0].size(0)
        crop_vbase= []

        for N_d in range(self.num_Ds):
            crop_base_imgs = torch.zeros(this_batch, 1, self.img_size, self.img_size)
            for step, (base_img_list) in enumerate(data[5]):

                temp_base_list = os.listdir(base_img_list)
                base_ix = random.randint(0, len(temp_base_list) - 1)
                base_img_name = '%s/%s.nii.gz' % (base_img_list, str(base_ix))

                base_img = nib.load(base_img_name)
                self.affine = base_img.affine
                base_img = np.array(base_img.dataobj)
                base_img = base_img.astype(np.float64)

                crop_base = resize(base_img, (self.img_size, self.img_size))
                crop_base = torch.from_numpy(crop_base)
                crop_base = crop_base.type(torch.FloatTensor)
                crop_base = torch.clamp(crop_base, min=-1000, max=500)
                crop_base = torch.add(crop_base, 1000)
                crop_base = torch.unsqueeze(crop_base, 0)
                crop_base = transforms.Normalize(mean=[750], std=[750])(crop_base)

                crop_base_imgs[step, :] = crop_base

            if CFG_CUDA:
                crop_vbase.append(Variable(crop_base_imgs).cuda())
            else:
                crop_vbase.append(Variable(crop_base_imgs))

        real_vimgs, real_vsegs, wrong_vimgs, wrong_vsegs = [], [], [], []
        if CFG_CUDA:
            vembedding = Variable(t_embedding).cuda()
        else:
            vembedding = Variable(t_embedding)

        for i in range(self.num_Ds):
            if CFG_CUDA:
                real_vimgs.append(Variable(imgs[i]).cuda())
                real_vsegs.append(Variable(segs[i]).cuda())
                wrong_vimgs.append(Variable(w_imgs[i]).cuda())
                wrong_vsegs.append(Variable(w_segs[i]).cuda())
            else:
                real_vimgs.append(Variable(imgs[i]))
                real_vsegs.append(Variable(segs[i]))
                wrong_vimgs.append(Variable(w_imgs[i]))
                wrong_vsegs.append(Variable(w_segs[i]))

        return real_vimgs, real_vsegs, wrong_vimgs, wrong_vsegs, vembedding, crop_vbase

    def train_Dnet(self, idx):

        batch_size = self.real_imgs[idx].size(0)
        criterion = self.criterion
        c_code = self.c_code

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        wrong_imgs = self.wrong_imgs[idx]

        # Forward
        netD.zero_grad()
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        errD = 0

        fake_imgs = self.fake_imgs[idx]
        real_segs = self.real_segs[idx]
        wrong_segs = self.wrong_segs[idx]
        fake_segs = self.fake_segs[idx]
        # Discriminant

        real_logits = netD(real_imgs, c_code.detach(), real_segs)
        wtxt_logits = netD(wrong_imgs, c_code.detach(), wrong_segs)
        wseg_logits = netD(real_imgs, c_code.detach(), wrong_segs)
        fake_logits = netD(fake_imgs.detach(), c_code.detach(), fake_segs.detach())

        # img-seg-txt
        errD_real = CFG_TRAIN_COEFF_ITS_LOSS*criterion(real_logits[0], real_labels)
        errD_wtxt = CFG_TRAIN_COEFF_ITS_LOSS*criterion(wtxt_logits[0], fake_labels)
        errD_wseg = CFG_TRAIN_COEFF_ITS_LOSS*criterion(wseg_logits[0], fake_labels)
        errD_fake = CFG_TRAIN_COEFF_ITS_LOSS*criterion(fake_logits[0], fake_labels)

        # img?
        if len(real_logits) > 0 and CFG_TRAIN_COEFF_I_LOSS > 0:
            errD_real_uncond = CFG_TRAIN_COEFF_I_LOSS * \
                               criterion(real_logits[1], real_labels)
            errD_wtxt_uncond = CFG_TRAIN_COEFF_I_LOSS * \
                               criterion(wtxt_logits[1], real_labels)
            errD_wseg_uncond = CFG_TRAIN_COEFF_I_LOSS * \
                               criterion(wseg_logits[1], real_labels)
            errD_fake_uncond = CFG_TRAIN_COEFF_I_LOSS * \
                               criterion(fake_logits[1], fake_labels)

            #
            errD_real = errD_real + errD_real_uncond
            errD_wtxt = errD_wtxt + errD_wtxt_uncond
            errD_wseg = errD_wseg + errD_wseg_uncond
            errD_fake = errD_fake + errD_fake_uncond


        # img-seg pair
        if len(real_logits) > 0 and CFG_TRAIN_COEFF_IS_LOSS > 0:
            errD_real_seg = CFG_TRAIN_COEFF_IS_LOSS * \
                            criterion(real_logits[2], real_labels)
            errD_wtxt_seg = CFG_TRAIN_COEFF_IS_LOSS * \
                            criterion(wtxt_logits[2], real_labels)
            errD_wseg_seg = CFG_TRAIN_COEFF_IS_LOSS * \
                            criterion(wseg_logits[2], fake_labels)
            errD_fake_seg = CFG_TRAIN_COEFF_IS_LOSS * \
                            criterion(fake_logits[2], fake_labels)
            #
            errD_real = errD_real + errD_real_seg
            errD_wtxt = errD_wtxt + errD_wtxt_seg
            errD_wseg = errD_wseg + errD_wseg_seg
            errD_fake = errD_fake + errD_fake_seg
            #
            errD = errD + errD_real + errD_wtxt + errD_wseg + errD_fake

      # backward
        errD.backward()
        # update parameters
        optD.step()

        return errD


    def train_Gnet(self):
        self.netG.zero_grad()
        errG_total = 0
        errG_pair =0
        batch_size = self.real_imgs[0].size(0)
        criterion = self.criterion
        real_labels = self.real_labels[:batch_size]
        c_code = self.c_code

        for i in range(self.num_Ds):

            outputs = self.netsD[i](self.fake_imgs[i], c_code, self.fake_segs[i])

            errG = CFG_TRAIN_COEFF_ITS_LOSS*criterion(outputs[0], real_labels)
            errG_pair += errG

            if batch_size > 0 and CFG_TRAIN_COEFF_I_LOSS > 0:
                errG_patch = CFG_TRAIN_COEFF_I_LOSS * \
                             criterion(outputs[1], real_labels)
                errG = errG + errG_patch

            if batch_size > 0 and CFG_TRAIN_COEFF_IS_LOSS > 0:
                errG_seg = CFG_TRAIN_COEFF_IS_LOSS * \
                           criterion(outputs[2], real_labels)
                errG = errG + errG_seg

                errG_total = errG_total + errG

            if CFG_TRAIN_COEFF_RC_LOSS > 0:
                # img64 - only BG reconstruction
                temp_seg = (self.fake_segs[0].detach())
                BG_mask = (temp_seg < 0).type(torch.FloatTensor)

                erode_mask = BG_mask.permute(0, 2, 3, 1).data.cpu().numpy()
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size), np.uint8)

                for b_idx in range(batch_size):
                    erode_mask[b_idx, :, :, 0] = cv2.erode(erode_mask[b_idx, :, :, 0], kernel, iterations=1)

                BG_mask = Variable(torch.FloatTensor(erode_mask)).cuda()
                BG_mask = BG_mask.permute(0, 3, 1, 2)
                BG_fake = torch.mul(self.fake_imgs[i], BG_mask)
                BG_img = torch.mul(self.crop_base[i], BG_mask)

                err_BG = CFG_TRAIN_COEFF_RC_LOSS * self.RC_criterion(BG_fake, BG_img)

                FG_mask = (temp_seg > 0).type(torch.FloatTensor)
                FG_mask = FG_mask.permute(0, 2, 3, 1).data.cpu().numpy()
                for b_idx in range(batch_size):
                    FG_mask[b_idx, :, :, 0] = largest_object(FG_mask[b_idx, :, :, 0])
                FG_mask = np.squeeze(FG_mask)

                errG_total = errG_total + err_BG

        errG_total.backward()
        self.optimizerG.step()

        return errG_total, errG_pair, err_BG

    def train(self):
        self.netG, self.netsD, self.num_Ds, \
            start_count = load_network(self.gpus)

        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.MSELoss()
        self.RC_criterion = nn.L1Loss()

        self.real_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(0))

        nz = CFG_GAN_Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))

        if CFG_CUDA:
            self.criterion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        predictions = []
        count = start_count
        start_epoch = start_count // (self.num_batches)
        self.SNAPSHOT_INTERVAL = self.num_batches * 10
        print('save model each %i' %self.SNAPSHOT_INTERVAL)

        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.real_imgs, self.real_segs, self.wrong_imgs, self.wrong_segs, \
                self.txt_embedding, self.crop_base = self.prepare_data(data)
                # genome embedding
                embedding = self.txt_embedding.data
                embedding = torch.squeeze(embedding)
                if CFG_CUDA:
                    embedding = Variable(embedding).cuda()
                else:
                    embedding = Variable(embedding)

                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                # print('DEBUG: Training ForwardProp noise: %s, embedding: %s, base: %s' % (noise.shape, embedding.shape,self.crop_base[0].shape))
                self.c_code, self.fake_imgs, self.fake_segs, _ = self.netG(noise, embedding, self.crop_base[0])

                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i)
                    errD_total += errD

                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                errG_total, errG_pair, errG_BG = self.train_Gnet()
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                count = count + 1

                if count % self.SNAPSHOT_INTERVAL == 0 or True:
                    # Save model
                    # save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)

                    # Save embeddings
                    embedding_test = self.embedding_test
                    embedding_all = self.embedding_all
                    base_test = self.img_test
                    noise_test = torch.FloatTensor(len(embedding_test), nz).normal_(0, 1)
                    if CFG_CUDA:
                        noise_test = Variable(noise_test).cuda()
                        embedding_test = Variable(embedding_test).cuda()
                        embedding_all = Variable(embedding_all).cuda()
                        base_test = Variable(base_test).cuda()
                    else:
                        noise_test = Variable(noise_test)
                        embedding_test = Variable(embedding_test)
                        embedding_all = Variable(embedding_all)
                        base_test = Variable(base_test)

                    c_code = self.netG.module.ec_net(embedding_all)
                    c_code = c_code.data.cpu().numpy()
                    np.savetxt('{}/Embedding_ep{}.txt'.format(self.embed_dir, epoch + 1), c_code, fmt='%f')

                    # print('DEBUG: Trainer.py train. noise: %s, embedding: %s, base: %s' % (fixed_noise.shape, embedding_test.shape, base_test.shape))

                    _, self.fake_imgs, self.fake_segs, self.fg_switchs = self.netG(fixed_noise, embedding_test, base_test)

                    img = self.fake_imgs[-1].data
                    grid = make_grid(img)
                    ndarr = grid.mul_(750).add_(-250).clamp_(-1000, 1000).permute(2, 1, 0).cpu().numpy()
                    im_out = ndarr[:,:,0]
                    nib.save(nib.Nifti1Image(im_out, self.affine), '{}/Image/Image_ep{}.nii.gz'.format(self.output_dir, epoch + 1))

                    img = self.fg_switchs[-1].data
                    grid = make_grid(img)
                    ndarr = grid.mul_(100).permute(2, 1, 0).cpu().numpy()
                    im_out = ndarr[:,:,0]
                    nib.save(nib.Nifti1Image(im_out, self.affine), '{}/Image/Switch_ep{}.nii.gz'.format(self.output_dir, epoch + 1))

                    temp_seg = self.fake_segs[-1]
                    mask = (temp_seg > 0).type(torch.FloatTensor)
                    mask = mask.permute(0, 2, 3, 1).data.cpu().numpy()
                    for b_idx in range(self.batch_size):
                        mask[b_idx, :, :, 0] = largest_object(mask[b_idx, :, :, 0])
                    mask = torch.from_numpy(mask).permute(0, 3, 1, 2)
                    grid = make_grid(mask)
                    ndarr = grid.mul_(255).clamp_(0, 255).permute(2, 1, 0).cpu().numpy()
                    im_out = ndarr[:,:,0]
                    nib.save(nib.Nifti1Image(im_out, self.affine), '{}/Image/Seg_ep{}.nii.gz'.format(self.output_dir, epoch + 1))

                    grid = make_grid(base_test.data)
                    ndarr = grid.mul_(750).add_(-250).clamp_(-1000, 1000).permute(2, 1, 0).cpu().numpy()
                    im_out = ndarr[:,:,0]
                    nib.save(nib.Nifti1Image(im_out, self.affine), '{}/Image/BASE.nii.gz'.format(self.output_dir, epoch + 1))

                    load_params(self.netG, backup_para)

            end_t = time.time()
            print('''[%d/%d][%d]
            Loss_D: %.2f Loss_G: %.2f Loss_pair: %.2f Loss_BG: %.2f Time: %.2fs
                      '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.data, errG_total.data,
                     errG_pair.data, errG_BG.data, end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)


    def save_evalimages(self, images, segs, base_img, filename, save_dir):
        s_tmp = '%s/%s' % (save_dir, filename[0])
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)

        grid = make_grid(images.data)
        ndarr = grid.mul_(750).add_(-250).clamp_(-1000, 1000).permute(2, 1, 0).cpu().numpy()
        im_out = ndarr[:, :, 0]
        nib.save(nib.Nifti1Image(im_out, self.affine),
                 '{}_Image.nii.gz'.format(s_tmp))

        mask = (segs > 0).type(torch.FloatTensor)
        mask = mask.permute(0, 2, 3, 1).data.cpu().numpy()
        for b_idx in range(mask.shape[0]):
            mask[b_idx, :, :, 0] = largest_object(mask[b_idx, :, :, 0])
        mask = torch.from_numpy(mask).permute(0, 3, 1, 2)
        grid = make_grid(mask)
        segarr = grid.mul_(255).clamp_(0, 255).permute(2, 1, 0).cpu().numpy()
        im_out = segarr[:, :, 0]
        nib.save(nib.Nifti1Image(im_out, self.affine),
                 '{}_Seg.nii.gz'.format(s_tmp))

        grid = make_grid(base_img.data)
        basearr = grid.mul_(750).add_(-250).clamp_(-1000, 1000).permute(2, 1, 0).cpu().numpy()
        im_out = basearr[:, :, 0]
        nib.save(nib.Nifti1Image(im_out, self.affine),
                 '{}_Base.nii.gz'.format(s_tmp))


    def save_single_evalimages(self, images, filename, save_dir):
        folder = '%s/%s' % (save_dir, filename)
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)

        for image_id in range(images.data.shape[0]):
            s_tmp = '%s/%s' % (folder, image_id)
            image_single = images.data[image_id,:]
            image_single = image_single.mul_(750).add_(-250).clamp_(-1000, 1000).permute(1, 2, 0).cpu().numpy()
            image_single = image_single[:, :, 0]
            nib.save(nib.Nifti1Image(image_single, self.affine),'{}.nii.gz'.format(s_tmp))


    def evaluate(self):
        if CFG_TRAIN_NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            self.num_Ds = CFG_TREE_BRANCH_NUM
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)
            state_dict = torch.load(CFG_TRAIN_NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', CFG_TRAIN_NET_G)

            # switch to evaluate mode
            netG.eval()

            # the path to save generated images
            s_tmp = CFG_TRAIN_NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d' % (s_tmp, iteration)

            nz = CFG_GAN_Z_DIM

            if CFG_CUDA:
                netG.cuda()

            switch = 10

            if switch > 100:
                embedding_test = self.embedding_test
                base_test = self.img_test
                base_ref = self.img_ref
                base_test_single1 = base_test[3,:,:,:]
                base_test_single1 = torch.unsqueeze(base_test_single1, 0)
                base_test_single1 = base_test_single1.expand(len(embedding_test), base_test.shape[1], base_test.shape[2],base_test.shape[3])
                base_test_single2 = base_test[10,:,:,:]
                base_test_single2 = torch.unsqueeze(base_test_single2, 0)
                base_test_single2 = base_test_single2.expand(len(embedding_test), base_test.shape[1], base_test.shape[2],base_test.shape[3])

                noise_test1 = torch.FloatTensor(len(embedding_test), nz).normal_(0, 1)
                noise_test2 = torch.FloatTensor(len(embedding_test), nz).normal_(0, 1)

                if CFG_CUDA:
                    noise_test1 = Variable(noise_test1).cuda()
                    noise_test2 = Variable(noise_test2).cuda()
                    embedding_test = Variable(embedding_test).cuda()
                    base_test = Variable(base_test).cuda()
                    base_test_single1 = Variable(base_test_single1).cuda()
                    base_test_single2 = Variable(base_test_single2).cuda()
                    base_ref = Variable(base_ref).cuda()
                else:
                    noise_test1 = Variable(noise_test1)
                    noise_test2 = Variable(noise_test2)
                    embedding_test = Variable(embedding_test)
                    base_test = Variable(base_test)
                    base_test_single1 = Variable(base_test_single1)
                    base_test_single2 = Variable(base_test_single2)
                    base_ref = Variable(base_ref)

                _, fake_imgs, fake_segs, _ = netG(noise_test1, embedding_test, base_test)
                self.save_evalimages(fake_imgs[0], fake_segs[0], base_test, ['Test'], save_dir)

                _, fake_imgs, fake_segs, _ = netG(noise_test1, embedding_test, base_ref)
                self.save_evalimages(fake_imgs[0], fake_segs[0], base_ref, ['Ref'], save_dir)

                _, fake_imgs, fake_segs, _ = netG(noise_test1, embedding_test, base_test_single1)
                self.save_evalimages(fake_imgs[0], fake_segs[0], base_test_single1, ['TestSingle1N1'], save_dir)

                _, fake_imgs, fake_segs, _ = netG(noise_test1, embedding_test, base_test_single2)
                self.save_evalimages(fake_imgs[0], fake_segs[0], base_test_single2, ['TestSingle2N1'], save_dir)

                _, fake_imgs, fake_segs, _ = netG(noise_test2, embedding_test, base_test_single1)
                self.save_evalimages(fake_imgs[0], fake_segs[0], base_test_single1, ['TestSingle1N2'], save_dir)

                _, fake_imgs, fake_segs, _ = netG(noise_test2, embedding_test, base_test_single2)
                self.save_evalimages(fake_imgs[0], fake_segs[0], base_test_single2, ['TestSingle2N2'], save_dir)

            elif switch > 0:
                embedding_test = self.embedding_test
                base_test = self.img_test
                base_ref = self.img_ref

                for case_id in range(len(embedding_test)):
                    base_test_single = base_test[case_id, :, :, :]
                    base_test_single = torch.unsqueeze(base_test_single, 0)
                    base_ref_single = base_ref[case_id, :, :, :]
                    base_ref_single = torch.unsqueeze(base_ref_single, 0)
                    embedding_test_single = embedding_test[case_id, :]
                    embedding_test_single = torch.unsqueeze(embedding_test_single, 0)
                    noise_test_single = torch.FloatTensor(1, nz).normal_(0, 1)
                    if CFG_CUDA:
                        noise_test_single = Variable(noise_test_single).cuda()
                        embedding_test_single = Variable(embedding_test_single).cuda()
                        base_test_single = Variable(base_test_single).cuda()
                        base_ref_single = Variable(base_ref_single).cuda()
                    else:
                        noise_test_single = Variable(noise_test_single).cuda()
                        embedding_test_single = Variable(embedding_test_single).cuda()
                        base_test_single = Variable(base_test_single).cuda()
                        base_ref_single = Variable(base_ref_single).cuda()

                    _, fake_imgs, fake_segs, _ = netG(noise_test_single, embedding_test_single, base_test_single)
                    self.save_evalimages(fake_imgs[0], fake_segs[0], base_test_single, ['Test_' + str(case_id)],
                                         save_dir)

                    _, fake_imgs, fake_segs, _ = netG(noise_test_single, embedding_test_single, base_ref_single)
                    self.save_evalimages(fake_imgs[0], fake_segs[0], base_ref_single, ['Ref_' + str(case_id)], save_dir)


            else:
                # each base image folder contains 10 images, process first base_ct
                base_ct = 10
                noise_size = 1
                folder_processed = []
                for step, data in enumerate(self.data_loader):
                    imgs, t_embedding, filenames, folders = data

                    folder_to_process = folders[0]
                    embeddingSingle = t_embedding[0, :]
                    embedding = torch.zeros(noise_size * base_ct, t_embedding.shape[2])

                    noise_whole = torch.FloatTensor(noise_size * base_ct, nz)

                    if not folder_to_process in folder_processed:
                        folder_processed.append(folder_to_process)

                        # Specify noise
                        noise = torch.FloatTensor(noise_size, nz)
                        noise.normal_(0, 1)

                        # Add 8 images within folder
                        # Same embedding
                        vbase = []
                        base_imgs = torch.zeros(noise_size * base_ct, 1, self.img_size, self.img_size)
                        for base_ix in range(base_ct):
                            base_img_name = '%s/%s.nii.gz' % (folder_to_process, str(base_ix))

                            base_img = nib.load(base_img_name)
                            self.affine = base_img.affine
                            base_img = np.array(base_img.dataobj)
                            base_img = base_img.astype(np.float64)
                            base_img = resize(base_img, (self.img_size, self.img_size))
                            base_img = torch.from_numpy(base_img)
                            base_img = base_img.type(torch.FloatTensor)
                            base_img = torch.clamp(base_img, min=-1000, max=500)
                            base_img = torch.add(base_img, 1000)
                            base_img = torch.unsqueeze(base_img, 0)
                            base_img = transforms.Normalize(mean=[750], std=[750])(base_img)
                            for base_img_cp in range(noise_size):
                                base_idx = base_ix * noise_size + base_img_cp
                                base_imgs[base_idx, :] = base_img
                                embedding[base_idx, :] = embeddingSingle
                                noise_whole[base_idx, :] = noise[base_img_cp, :]

                        if CFG_CUDA:
                            base_imgs = Variable(base_imgs).cuda()
                            embedding = Variable(embedding).cuda()
                            noise_whole = Variable(noise_whole).cuda()
                        else:
                            base_imgs = Variable(base_imgs)
                            embedding = Variable(embedding)
                            noise_whole = Variable(noise_whole)

                        _, fake_imgs, fake_segs, _ = netG(noise_whole, embedding, base_imgs)
                        self.save_single_evalimages(fake_imgs[0], os.path.basename(folder_to_process), save_dir)
