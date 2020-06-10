from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms

import nibabel as nib
from skimage.transform import resize

import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
CFG_TREE_BRANCH_NUM = 1
CFG_TRAIN_FLAG = True

import torch

IMG_EXTENSIONS = ['.nii.gz']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(img_path, seg_name, imsize, bbox=None,
             transform=None):

    img = nib.load(img_path)
    # print('DEBUG: nib img load type: ', type(img))
    affine = img.affine
    img = np.array(img.dataobj)
    # print('DEBUG: img init shape: ', img.shape)
    img = img.astype(np.float64)

    seg = nib.load(seg_name)
    seg = np.array(seg.dataobj) > 0.5
    seg = seg.astype(np.float64)

    width, height = img.shape
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]))
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img[x1:x2, y1:y2]
        seg = seg[x1:x2, y1:y2]
    else:
        print('FOR SOME REASON BOUNDING BOX WAS NONE')

    ret_img, ret_seg = [], []
    for i in range(CFG_TREE_BRANCH_NUM):
        re_img = resize(img, (imsize[i], imsize[i]))
        re_seg = resize(seg, (imsize[i], imsize[i]))

        re_img = torch.from_numpy(re_img)
        re_img = re_img.type(torch.FloatTensor)
        re_img = torch.clamp(re_img, min=-1000, max=500)
        re_img = torch.add(re_img, 1000)
        re_img = torch.unsqueeze(re_img, 0)
        re_img = transforms.Normalize(mean=[750], std=[750])(re_img)

        re_seg = torch.from_numpy(re_seg)
        re_seg = re_seg.type(torch.FloatTensor)
        re_seg = torch.unsqueeze(re_seg, 0)
        re_seg = transforms.Normalize(mean=[0.5], std=[0.5])(re_seg)

        # ndarr = re_img.mul_(1000).clamp_(-1000, 1000).permute(2, 1, 0).cpu().numpy()
        # im_out = ndarr[:, :, 0]
        # nib.save(nib.Nifti1Image(im_out, affine),'Temp/img.nii.gz')
        #
        # ndarr = re_seg.mul_(255).clamp_(0, 255).permute(2, 1, 0).cpu().numpy()
        # im_out = ndarr[:, :, 0]
        # nib.save(nib.Nifti1Image(im_out, affine),'Temp/msk.nii.gz')

        ret_img.append(re_img)
        ret_seg.append(re_seg)

    return ret_img, ret_seg

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64, transform=None):
        self.transform = transform

        self.imsize = []
        self.imsize.append(base_size)

        self.data = []
        self.data_dir = data_dir
        # print('---------------------',data_dir)
        self.bbox = self.load_bbox()

        split_dir = os.path.join(data_dir, 'infos/')
        # print(split_dir)
        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))

        if CFG_TRAIN_FLAG:
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'infos/bounding_boxes.txt')

        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'infos/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        # print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-7]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-7]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_embedding(self, data_dir):

        embedding_filename = '/RNA.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding='latin1')
            embeddings = np.array(embeddings)
            embeddings = np.float32(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            # print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        # print(filepath)
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        # print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames


    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        embedding = self.embeddings[index, :]

        # print('single embedding', embeddings.shape)

        img_name = '%s/images/%s.nii.gz' % (data_dir, key)
        seg_name = '%s/segmentations/%s.nii.gz' % (data_dir, key)
        (imgs, segs) = get_imgs(img_name, seg_name, self.imsize,
                        bbox, self.transform)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if(self.class_id[index] == self.class_id[wrong_ix]):
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
        else:
            wrong_bbox = None
        wrong_img_name = '%s/images/%s.nii.gz' % (data_dir, wrong_key)
        wrong_seg_name = '%s/segmentations/%s.nii.gz' %(data_dir, wrong_key)
        wrong_imgs, wrong_segs = get_imgs(wrong_img_name, wrong_seg_name, self.imsize,
                                        wrong_bbox, self.transform)

        base_key = key.split('/')
        base_img_name = '%s/base_images/%s' % (data_dir, base_key[0])

        # for item in [imgs, segs, wrong_imgs, wrong_segs, embedding, base_img_name]:
        #     print(type(item))

        # print(imgs[0].shape)

        return imgs, segs, wrong_imgs, wrong_segs, embedding, base_img_name

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        embedding = self.embeddings[index, :]
        img_name = '%s/images/%s.nii.gz' % (data_dir, key)
        seg_name = '%s/segmentations/%s.nii.gz' % (data_dir, key)
        imgs, _ = get_imgs(img_name, seg_name, self.imsize,
                        bbox, self.transform)

        base_key = key.split('/')
        base_img_name = '%s/base_images/%s' % (data_dir, base_key[0])

        return imgs, embedding, key, base_img_name

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)
