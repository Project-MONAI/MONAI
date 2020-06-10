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

import time
import datetime
import dateutil.tz

import torch
import numpy as np

from monai import config
from monai.data import DataLoader
from monai.utils.misc import set_determinism
from monai.utils.enums import InterpolateMode
from monai.transforms import (Compose, 
                              LoadNiftid,
                              Resized,
                              AddChanneld,
                              ThresholdIntensityd,
                              ShiftIntensityd,
                              NormalizeIntensityd,
                              ToTensord,
                              Lambdad
                              )

from DataPreprocessor import load_rg_data
from Dataset import RGDataset
from Transforms import CropWithBoundingBoxd, ClipIntensityd
# import torchvision.transforms as transforms

from src.datasets import TextDataset as OGDataset
from src.trainer import condGANTrainer as trainer

def main():
    config.print_config()

    # init seed 
    set_determinism(12345)

    # define output dir 
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    output_data_dir = 'ModelOut/radiogenomic-gan_%s' % (now.strftime('%Y_%m_%d_%H_%M_%S'))

    # define input directory
    input_data_dir = '/home/gagan/code/MONAI/research/radiogenomic-gan-nodule-synthesis/data'
    
    print('Input Data Dir: %s' % input_data_dir)
    print('Output Data Dir: %s' % output_data_dir)

    # load data dictionary using Data Preprocessor
    data_dict = load_rg_data(input_data_dir)

    # define transforms for image  
    image_size = 128
    image_shape = [image_size, image_size]

    train_transforms = Compose([
        LoadNiftid(keys=['image', 'seg'], dtype=np.float64),
        CropWithBoundingBoxd(keys=['image', 'seg'], bbox_key='bbox'),
        AddChanneld(keys=['image', 'seg']),
        ThresholdIntensityd(keys=['seg'], threshold=0.5),
        Resized(keys=['image', 'seg'], spatial_size=image_shape, mode=InterpolateMode.AREA),
        ThresholdIntensityd(keys=['image'], threshold=-1000, above=True, cval=-1000),
        ThresholdIntensityd(keys=['image'], threshold=500, above=False, cval=500),
        ShiftIntensityd(keys=['image'], offset=1000),
        NormalizeIntensityd(keys=['image'], 
                            subtrahend=np.full(image_shape, 750.0),
                            divisor=np.full(image_shape, 750.0),
                            channel_wise=True),
        NormalizeIntensityd(keys=['seg'], 
                            subtrahend=np.full(image_shape, 0.5),
                            divisor=np.full(image_shape, 0.5),
                            channel_wise=True),
        ToTensord(keys=['image', 'seg'])
    ])

    # create dataset
    cache_max: int = 25
    dataset = RGDataset(data_dict, train_transforms, cache_max)
    # dataset = OGDataset(input_data_dir, 'train', base_size=128, transform=None)

    # define DataLoader()
    batch_size = 16
    num_gpu = 1
    num_workers: int = 5

    dataloader = DataLoader(
        dataset, batch_size=batch_size * num_gpu,
        drop_last=True, shuffle=False, num_workers=num_workers)

    # create GAN network
    algo = trainer(output_data_dir, dataloader, image_size, input_data_dir)
    print('created trainer: ', algo)

    # create optimizer and loss

    # start training
    start_t = time.time()
    algo.train()
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
    
    # save    
    print('END MAIN')


if __name__ == "__main__":
    main()

