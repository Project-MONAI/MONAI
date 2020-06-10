from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'nodules'
__C.EMBEDDING_TYPE = 'rna'
__C.CONFIG_NAME = 'MC_GAN'
__C.DATA_DIR = ''

__C.GPU_ID = '0'
__C.CUDA = True

__C.WORKERS = 4

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 1
__C.TREE.BASE_SIZE = 128


# Test options
__C.TEST = edict()
__C.TEST.B_EXAMPLE = False
__C.TEST.SAMPLE_NUM = 30000


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.VIS_COUNT = 64
__C.TRAIN.MAX_EPOCH = 1500
__C.TRAIN.SNAPSHOT_INTERVAL = 5000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.FLAG = False
__C.TRAIN.NET_G = '../examples/birds_MC_GAN_2018_07_19_03_26_27/Model/netG_414000.pth'
__C.TRAIN.NET_D = ''

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0
__C.TRAIN.COEFF.ITS_LOSS = 2.0
__C.TRAIN.COEFF.I_LOSS = 2.0
__C.TRAIN.COEFF.IS_LOSS = 1.0
__C.TRAIN.COEFF.RC_LOSS = 15.0
__C.TRAIN.COEFF.AR_LOSS = 25.0

# Modal options
__C.GAN = edict()
__C.GAN.EMBEDDING_DIM = 128
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 64
__C.GAN.Z_DIM = 100
__C.GAN.NETWORK_TYPE = 'default'


__C.TEXT = edict()
__C.TEXT.DIMENSION = 5268


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

