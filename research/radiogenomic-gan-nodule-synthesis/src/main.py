from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time

from MyTransform import *
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
from miscc.config import cfg, cfg_from_file

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',help='optional config file',
                        default='cfg/nodule_3stages.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    print('***********', cfg.CONFIG_NAME)
    print('Using config:')
    pprint.pprint(cfg)
    print('***********', cfg.DATA_DIR)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = 'ModelOut/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        if cfg.DATASET_NAME == 'nodules':
            bshuffle = False
            split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    image_transform = transforms.Compose([ Rescale(int(imsize * 66 / 64)),
        RandomCrop(imsize),RandomHorizontalFlip()])

    #image_transform = transforms.Compose([Rescale(int(imsize )), RandomHorizontalFlip()])

    from datasets import TextDataset

    print('****************', cfg.DATA_DIR)

    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    assert dataset
    num_gpu = len(cfg.GPU_ID.split(','))
    print('GPU count ', num_gpu)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate

    from trainer import condGANTrainer as trainer
    algo = trainer(output_dir, dataloader, imsize, cfg.DATA_DIR)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.evaluate()
    end_t = time.time()
    print('Total time for training:', end_t - start_t)

