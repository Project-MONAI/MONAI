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
The predefined properties for a bundle workflow, other applications can leverage the properties
to interact with the bundle workflow.
Some properties are required and some are optional, optional properties mean: if some component of the
bundle workflow refer to the property, the property must be defined, otherwise, the property can be None.
Every item in this `TrainProperties` or `InferProperties` or `MetaProperties` dictionary is a property,
the key is the property name and the values include:
1. description.
2. whether it's a required property.
3. config item ID name (only applicable when the bundle workflow is defined in config).
4. reference config item ID name (only applicable when the bundle workflow is defined in config).

"""

from __future__ import annotations

from monai.bundle.utils import ID_SEP_KEY
from monai.utils import BundleProperty, BundlePropertyConfig

TrainProperties = {
    "bundle_root": {
        BundleProperty.DESC: "root path of the bundle.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "bundle_root",
    },
    "device": {
        BundleProperty.DESC: "target device to execute the bundle workflow.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "device",
    },
    "dataset_dir": {
        BundleProperty.DESC: "directory path of the dataset.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "dataset_dir",
    },
    "trainer": {
        BundleProperty.DESC: "training workflow engine.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: f"train{ID_SEP_KEY}trainer",
    },
    "network_def": {
        BundleProperty.DESC: "network module for the training.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "network_def",
    },
    "max_epochs": {
        BundleProperty.DESC: "max number of epochs to execute the training.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: f"train{ID_SEP_KEY}trainer{ID_SEP_KEY}max_epochs",
    },
    "train_dataset": {
        BundleProperty.DESC: "PyTorch dataset object for the training logic.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: f"train{ID_SEP_KEY}dataset",
    },
    "train_inferer": {
        BundleProperty.DESC: "MONAI Inferer object to execute the model computation in training.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: f"train{ID_SEP_KEY}inferer",
    },
    "train_dataset_data": {
        BundleProperty.DESC: "data source for the training dataset.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"train{ID_SEP_KEY}dataset{ID_SEP_KEY}data",
        BundlePropertyConfig.REF_ID: None,  # no reference to this ID
    },
    "train_handlers": {
        BundleProperty.DESC: "event-handlers for the training logic.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"train{ID_SEP_KEY}handlers",
        BundlePropertyConfig.REF_ID: f"train{ID_SEP_KEY}trainer{ID_SEP_KEY}train_handlers",
    },
    "train_preprocessing": {
        BundleProperty.DESC: "preprocessing for the training input data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"train{ID_SEP_KEY}preprocessing",
        BundlePropertyConfig.REF_ID: f"train{ID_SEP_KEY}dataset{ID_SEP_KEY}transform",
    },
    "train_postprocessing": {
        BundleProperty.DESC: "postprocessing for the training model output data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"train{ID_SEP_KEY}postprocessing",
        BundlePropertyConfig.REF_ID: f"train{ID_SEP_KEY}trainer{ID_SEP_KEY}postprocessing",
    },
    "train_key_metric": {
        BundleProperty.DESC: "key metric to compute on the training data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"train{ID_SEP_KEY}key_metric",
        BundlePropertyConfig.REF_ID: f"train{ID_SEP_KEY}trainer{ID_SEP_KEY}key_train_metric",
    },
    "evaluator": {
        BundleProperty.DESC: "validation workflow engine.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"validate{ID_SEP_KEY}evaluator",
        BundlePropertyConfig.REF_ID: "validator",  # this REF_ID is the arg name of `ValidationHandler`
    },
    "val_interval": {
        BundleProperty.DESC: "validation interval during the training.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "val_interval",
        BundlePropertyConfig.REF_ID: "interval",  # this REF_ID is the arg name of `ValidationHandler`
    },
    "val_handlers": {
        BundleProperty.DESC: "event-handlers for the validation logic.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"validate{ID_SEP_KEY}handlers",
        BundlePropertyConfig.REF_ID: f"validate{ID_SEP_KEY}evaluator{ID_SEP_KEY}val_handlers",
    },
    "val_dataset": {
        BundleProperty.DESC: "PyTorch dataset object for the validation logic.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"validate{ID_SEP_KEY}dataset",
        BundlePropertyConfig.REF_ID: f"validate{ID_SEP_KEY}dataloader{ID_SEP_KEY}dataset",
    },
    "val_dataset_data": {
        BundleProperty.DESC: "data source for the validation dataset.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"validate{ID_SEP_KEY}dataset{ID_SEP_KEY}data",
        BundlePropertyConfig.REF_ID: None,  # no reference to this ID
    },
    "val_inferer": {
        BundleProperty.DESC: "MONAI Inferer object to execute the model computation in validation.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"validate{ID_SEP_KEY}inferer",
        BundlePropertyConfig.REF_ID: f"validate{ID_SEP_KEY}evaluator{ID_SEP_KEY}inferer",
    },
    "val_preprocessing": {
        BundleProperty.DESC: "preprocessing for the validation input data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"validate{ID_SEP_KEY}preprocessing",
        BundlePropertyConfig.REF_ID: f"validate{ID_SEP_KEY}dataset{ID_SEP_KEY}transform",
    },
    "val_postprocessing": {
        BundleProperty.DESC: "postprocessing for the validation model output data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"validate{ID_SEP_KEY}postprocessing",
        BundlePropertyConfig.REF_ID: f"validate{ID_SEP_KEY}evaluator{ID_SEP_KEY}postprocessing",
    },
    "val_key_metric": {
        BundleProperty.DESC: "key metric to compute on the validation data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"validate{ID_SEP_KEY}key_metric",
        BundlePropertyConfig.REF_ID: f"validate{ID_SEP_KEY}evaluator{ID_SEP_KEY}key_val_metric",
    },
}

InferProperties = {
    "bundle_root": {
        BundleProperty.DESC: "root path of the bundle.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "bundle_root",
    },
    "device": {
        BundleProperty.DESC: "target device to execute the bundle workflow.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "device",
    },
    "dataset_dir": {
        BundleProperty.DESC: "directory path of the dataset.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "dataset_dir",
    },
    "dataset": {
        BundleProperty.DESC: "PyTorch dataset object for the inference / evaluation logic.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "dataset",
    },
    "evaluator": {
        BundleProperty.DESC: "inference / evaluation workflow engine.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "evaluator",
    },
    "network_def": {
        BundleProperty.DESC: "network module for the inference.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "network_def",
    },
    "inferer": {
        BundleProperty.DESC: "MONAI Inferer object to execute the model computation in inference.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "inferer",
    },
    "dataset_data": {
        BundleProperty.DESC: "data source for the inference / evaluation dataset.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"dataset{ID_SEP_KEY}data",
        BundlePropertyConfig.REF_ID: None,  # no reference to this ID
    },
    "handlers": {
        BundleProperty.DESC: "event-handlers for the inference / evaluation logic.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "handlers",
        BundlePropertyConfig.REF_ID: f"evaluator{ID_SEP_KEY}val_handlers",
    },
    "preprocessing": {
        BundleProperty.DESC: "preprocessing for the input data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "preprocessing",
        BundlePropertyConfig.REF_ID: f"dataset{ID_SEP_KEY}transform",
    },
    "postprocessing": {
        BundleProperty.DESC: "postprocessing for the model output data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "postprocessing",
        BundlePropertyConfig.REF_ID: f"evaluator{ID_SEP_KEY}postprocessing",
    },
    "key_metric": {
        BundleProperty.DESC: "the key metric during evaluation.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "key_metric",
        BundlePropertyConfig.REF_ID: f"evaluator{ID_SEP_KEY}key_val_metric",
    },
}

MetaProperties = {
    "version": {
        BundleProperty.DESC: "bundle version",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: f"_meta_{ID_SEP_KEY}version",
    },
    "monai_version": {
        BundleProperty.DESC: "required monai version used for bundle",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: f"_meta_{ID_SEP_KEY}monai_version",
    },
    "pytorch_version": {
        BundleProperty.DESC: "required pytorch version used for bundle",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: f"_meta_{ID_SEP_KEY}pytorch_version",
    },
    "numpy_version": {
        BundleProperty.DESC: "required numpy version used for bundle",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: f"_meta_{ID_SEP_KEY}numpy_version",
    },
    "description": {
        BundleProperty.DESC: "description for bundle",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"_meta_{ID_SEP_KEY}description",
    },
    "spatial_shape": {
        BundleProperty.DESC: "spatial shape for the inputs",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"_meta_{ID_SEP_KEY}network_data_format{ID_SEP_KEY}inputs{ID_SEP_KEY}image"
        f"{ID_SEP_KEY}spatial_shape",
    },
    "channel_def": {
        BundleProperty.DESC: "channel definition for the prediction",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: f"_meta_{ID_SEP_KEY}network_data_format{ID_SEP_KEY}outputs{ID_SEP_KEY}pred{ID_SEP_KEY}channel_def",
    },
}
