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

import os

import numpy as np

from monai.transforms import DivisiblePad

STRUCTURES = (
    "BrainStem",
    "Chiasm",
    "Mandible",
    "OpticNerve_L",
    "OpticNerve_R",
    "Parotid_L",
    "Parotid_R",
    "Submandibular_L",
    "Submandibular_R",
)


def get_filenames(path, maskname=STRUCTURES):
    """
    create file names according to the predefined folder structure.

    Args:
        path: data folder name
        maskname: target structure names
    """
    maskfiles = []
    for seg in maskname:
        if os.path.exists(os.path.join(path, "./structures/" + seg + "_crp_v2.npy")):
            maskfiles.append(os.path.join(path, "./structures/" + seg + "_crp_v2.npy"))
        else:
            # the corresponding mask is missing seg, path.split("/")[-1]
            maskfiles.append(None)
    return os.path.join(path, "img_crp_v2.npy"), maskfiles


def load_data_and_mask(data, mask_data):
    """
    Load data filename and mask_data (list of file names)
    into a dictionary of {'image': array, "label": list of arrays, "name": str}.
    """
    pad_xform = DivisiblePad(k=32)
    img = np.load(data)  # z y x
    img = pad_xform(img[None])[0]
    item = dict(image=img, label=[])
    for maskfnm in mask_data:
        if maskfnm is None:
            ms = np.zeros(img.shape, np.uint8)
        else:
            ms = np.load(maskfnm).astype(np.uint8)
            assert ms.min() == 0 and ms.max() == 1
        mask = pad_xform(ms[None])[0]
        item["label"].append(mask)
    assert len(item["label"]) == 9
    item["name"] = str(data)
    return item
