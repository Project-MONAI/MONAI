# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of the remote MMAR descriptors

See Also:
    - https://docs.nvidia.com/clara/clara-train-sdk/pt/mmar.html
"""

import os

__all__ = ["MODEL_DESC", "RemoteMMARKeys"]


class RemoteMMARKeys:
    """
    Data keys used for loading MMAR.
    ID must uniquely define an MMAR.
    """

    ID = "id"  # unique MMAR
    NAME = "name"  # MMAR name for readability
    URL = "url"  # remote location of the MMAR
    DOC = "doc"  # documentation page of the remote model
    FILE_TYPE = "file_type"  # type of the compressed MMAR
    HASH_TYPE = "hash_type"  # hashing method for the compressed MMAR
    HASH_VAL = "hash_val"  # hashing value for the compressed MMAR
    MODEL_FILE = "model_file"  # within an MMAR folder, the relative path to the model file


MODEL_DESC = (
    {
        RemoteMMARKeys.ID: "clara_pt_prostate_mri_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_prostate_mri_segmentation",
        RemoteMMARKeys.URL: "https://api.ngc.nvidia.com/v2/models/nvidia/"
        "med/clara_pt_prostate_mri_segmentation/versions/1/zip",
        RemoteMMARKeys.DOC: "https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_prostate_mri_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
    },
    {
        RemoteMMARKeys.ID: "clara_pt_covid19_ct_lesion_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_covid19_ct_lesion_segmentation",
        RemoteMMARKeys.URL: "https://api.ngc.nvidia.com/v2/models/nvidia/"
        "med/clara_pt_covid19_ct_lesion_segmentation/versions/1/zip",
        RemoteMMARKeys.DOC: "https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_covid19_ct_lesion_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
    },
    {
        RemoteMMARKeys.ID: "clara_pt_fed_learning_brain_tumor_mri_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_fed_learning_brain_tumor_mri_segmentation",
        RemoteMMARKeys.URL: "https://api.ngc.nvidia.com/v2/models/nvidia/"
        "med/clara_pt_fed_learning_brain_tumor_mri_segmentation/versions/1/zip",
        RemoteMMARKeys.DOC: "https://ngc.nvidia.com/catalog/models/"
        "nvidia:med:clara_pt_fed_learning_brain_tumor_mri_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "server", "best_FL_global_model.pt"),
    },
)
