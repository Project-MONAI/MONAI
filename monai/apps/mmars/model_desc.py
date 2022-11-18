# Copyright (c) MONAI Consortium
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
from typing import Any, Dict, Tuple

__all__ = ["MODEL_DESC", "RemoteMMARKeys"]


class RemoteMMARKeys:
    """
    Data keys used for loading MMAR.
    ID must uniquely define an MMAR.
    """

    ID = "id"  # unique MMAR
    NAME = "name"  # MMAR name for readability
    URL = "url"  # remote location of the MMAR, see also: `monai.apps.mmars.mmars._get_ngc_url`
    DOC = "doc"  # documentation page of the remote model, see also: `monai.apps.mmars.mmars._get_ngc_doc_url`
    FILE_TYPE = "file_type"  # type of the compressed MMAR
    HASH_TYPE = "hash_type"  # hashing method for the compressed MMAR
    HASH_VAL = "hash_val"  # hashing value for the compressed MMAR
    MODEL_FILE = "model_file"  # within an MMAR folder, the relative path to the model file
    CONFIG_FILE = "config_file"  # within an MMAR folder, the relative path to the config file (for model config)
    VERSION = "version"  # version of the MMAR


MODEL_DESC: Tuple[Dict[Any, Any], ...] = (
    {
        RemoteMMARKeys.ID: "clara_pt_spleen_ct_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_spleen_ct_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_prostate_mri_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_prostate_mri_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_covid19_ct_lesion_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_covid19_ct_lesion_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_covid19_3d_ct_classification_1",
        RemoteMMARKeys.NAME: "clara_pt_covid19_3d_ct_classification",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_covid19_ct_lung_annotation_1",
        RemoteMMARKeys.NAME: "clara_pt_covid19_ct_lung_annotation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_fed_learning_brain_tumor_mri_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_fed_learning_brain_tumor_mri_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "server", "best_FL_global_model.pt"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_pathology_metastasis_detection_1",
        RemoteMMARKeys.NAME: "clara_pt_pathology_metastasis_detection",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_brain_mri_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_brain_mri_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_brain_mri_segmentation_t1c_1",
        RemoteMMARKeys.NAME: "clara_pt_brain_mri_segmentation_t1c",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_liver_and_tumor_ct_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_liver_and_tumor_ct_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_pancreas_and_tumor_ct_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_pancreas_and_tumor_ct_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_brain_mri_annotation_t1c_1",
        RemoteMMARKeys.NAME: "clara_pt_brain_mri_annotation_t1c",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_spleen_ct_annotation_1",
        RemoteMMARKeys.NAME: "clara_pt_spleen_ct_annotation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_deepgrow_3d_annotation_1",
        RemoteMMARKeys.NAME: "clara_pt_deepgrow_3d_annotation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_deepgrow_2d_annotation_1",
        RemoteMMARKeys.NAME: "clara_pt_deepgrow_2d_annotation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_covid19_ct_lung_segmentation_1",
        RemoteMMARKeys.NAME: "clara_pt_covid19_ct_lung_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_unetr_ct_btcv_segmentation",
        RemoteMMARKeys.NAME: "clara_pt_unetr_ct_btcv_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 4.1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_chest_xray_classification",
        RemoteMMARKeys.NAME: "clara_pt_chest_xray_classification",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models", "model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 4.1,
    },
    {
        RemoteMMARKeys.ID: "clara_pt_self_supervised_learning_segmentation",
        RemoteMMARKeys.NAME: "clara_pt_self_supervised_learning_segmentation",
        RemoteMMARKeys.FILE_TYPE: "zip",
        RemoteMMARKeys.HASH_TYPE: "md5",
        RemoteMMARKeys.HASH_VAL: None,
        RemoteMMARKeys.MODEL_FILE: os.path.join("models_2gpu", "best_metric_model.pt"),
        RemoteMMARKeys.CONFIG_FILE: os.path.join("config", "config_train.json"),
        RemoteMMARKeys.VERSION: 4.1,
    },
)
