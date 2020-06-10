# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import csv
import pickle
import numpy as np


def load_rg_data(data_dir):
    image_keys = _load_pickle(data_dir, 'filenames.pickle')
    class_info = _load_pickle(data_dir, 'class_info.pickle')
    embeddings = np.float32(_load_pickle(data_dir, 'RNA.pickle'))
    bboxes = _load_bounding_boxes(data_dir)
    data = _build_data_dict(data_dir, image_keys, bboxes, embeddings, class_info)
    return data


def _build_data_dict(data_dir, image_keys, bboxes, embeddings, class_info):
    data_samples = []
    for index, image_key in enumerate(image_keys):
        img_name = os.path.join(data_dir, 'images', '%s.nii.gz' % image_key)
        seg_name = os.path.join(data_dir, 'segmentations', '%s.nii.gz' % image_key)
        base_img_name = os.path.join(data_dir, 'base_images', '%s' % image_key.split('/')[0])
        data_sample = {}
        data_sample['image'] = img_name
        data_sample['seg'] = seg_name
        data_sample['base'] = base_img_name
        data_sample['bbox'] = bboxes[index]
        data_sample['embedding'] = embeddings[index, :]
        data_sample['class'] = class_info[index]
        data_samples.append(data_sample)
    return data_samples


def _load_bounding_boxes(data_dir):
    # bbox = [x-left, y-top, width, height]
    bbox_filename = os.path.join(data_dir, 'infos', 'bounding_boxes.txt')
    bboxes = []
    with open(bbox_filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            bbox_values = np.asarray(row).astype(int)
            bboxes.append(bbox_values[1:])   # ignore index value
    return bboxes


def _load_pickle(data_dir, pkl_name):
    pkl_path = os.path.join(data_dir, 'infos', pkl_name)
    file = open(pkl_path, 'rb')
    data = pickle.load(file, encoding='latin1')
    return np.array(data)
