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
Every item in this `TrainProperties` or `InferProperties` dictionary is a property,
the key is the property name and the values include:
1. description.
2. whether it's a required property.
3. config item ID name (only applicable when the bundle workflow is defined in config).
4. reference config item ID name (only applicable when the bundle workflow is defined in config).

"""

from __future__ import annotations

from monai.utils import BundleProperty, BundlePropertyConfig

TrainProperties = {
    "bundle_root": {
        BundleProperty.DESP: "root path of the bundle.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "bundle_root",
    },
    "device": {
        BundleProperty.DESP: "target device to execute the bundle workflow.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "device",
    },
    "dataset_dir": {
        BundleProperty.DESP: "directory path of the dataset.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "dataset_dir",
    },
    "trainer": {
        BundleProperty.DESP: "training workflow engine.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "train#trainer",
    },
    "max_epochs": {
        BundleProperty.DESP: "max number of epochs to execute the training.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "train#trainer#max_epochs",
    },
    "train_dataset": {
        BundleProperty.DESP: "PyTorch dataset object for the training logic.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "train#dataset",
    },
    "train_dataset_data": {
        BundleProperty.DESP: "data source for the training dataset.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "train#dataset#data",
    },
    "train_inferer": {
        BundleProperty.DESP: "MONAI Inferer object to execute the model computation in training.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "train#inferer",
    },
    "train_handlers": {
        BundleProperty.DESP: "event-handlers for the training logic.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "train#handlers",
        BundlePropertyConfig.REF_ID: "train#trainer#train_handlers",
    },
    "train_preprocessing": {
        BundleProperty.DESP: "preprocessing for the training input data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "train#preprocessing",
        BundlePropertyConfig.REF_ID: "train#dataset#transform",
    },
    "train_postprocessing": {
        BundleProperty.DESP: "postprocessing for the training model output data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "train#postprocessing",
        BundlePropertyConfig.REF_ID: "train#trainer#postprocessing",
    },
    "train_key_metric": {
        BundleProperty.DESP: "key metric to compute on the training data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "train#key_metric",
        BundlePropertyConfig.REF_ID: "train#trainer#key_train_metric",
    },
    "evaluator": {
        BundleProperty.DESP: "validation workflow engine.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "validate#evaluator",
        BundlePropertyConfig.REF_ID: "validator",  # this REF_ID is the arg name of `ValidationHandler`
    },
    "val_interval": {
        BundleProperty.DESP: "validation interval during the training.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "val_interval",
        BundlePropertyConfig.REF_ID: "interval",  # this REF_ID is the arg name of `ValidationHandler`
    },
    "val_handlers": {
        BundleProperty.DESP: "event-handlers for the validation logic.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "validate#handlers",
        BundlePropertyConfig.REF_ID: "validate#evaluator#val_handlers",
    },
    "val_dataset": {
        BundleProperty.DESP: "PyTorch dataset object for the validation logic.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "validate#dataset",
        BundlePropertyConfig.REF_ID: "validate#dataloader#dataset",
    },
    "val_dataset_data": {
        BundleProperty.DESP: "data source for the validation dataset.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "validate#dataset#data",
        BundlePropertyConfig.REF_ID: None,  # no reference to this ID
    },
    "val_inferer": {
        BundleProperty.DESP: "MONAI Inferer object to execute the model computation in validation.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "validate#inferer",
        BundlePropertyConfig.REF_ID: "validate#evaluator#inferer",
    },
    "val_preprocessing": {
        BundleProperty.DESP: "preprocessing for the validation input data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "validate#preprocessing",
        BundlePropertyConfig.REF_ID: "validate#dataset#transform",
    },
    "val_postprocessing": {
        BundleProperty.DESP: "postprocessing for the validation model output data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "validate#postprocessing",
        BundlePropertyConfig.REF_ID: "validate#evaluator#postprocessing",
    },
    "val_key_metric": {
        BundleProperty.DESP: "key metric to compute on the validation data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "validate#key_metric",
        BundlePropertyConfig.REF_ID: "validate#evaluator#key_val_metric",
    },
}


InferProperties = {
    "bundle_root": {
        BundleProperty.DESP: "root path of the bundle.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "bundle_root",
    },
    "device": {
        BundleProperty.DESP: "target device to execute the bundle workflow.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "device",
    },
    "network_def": {
        BundleProperty.DESP: "network module for the inference.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "network_def",
    },
    "inferer": {
        BundleProperty.DESP: "MONAI Inferer object to execute the model computation in inference.",
        BundleProperty.REQUIRED: True,
        BundlePropertyConfig.ID: "inferer",
    },
    "preprocessing": {
        BundleProperty.DESP: "preprocessing for the input data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "preprocessing",
        BundlePropertyConfig.REF_ID: "dataset#transform",
    },
    "postprocessing": {
        BundleProperty.DESP: "postprocessing for the model output data.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "postprocessing",
        BundlePropertyConfig.REF_ID: "evaluator#postprocessing",
    },
    "key_metric": {
        BundleProperty.DESP: "the key metric during evaluation.",
        BundleProperty.REQUIRED: False,
        BundlePropertyConfig.ID: "key_metric",
        BundlePropertyConfig.REF_ID: "evaluator#key_val_metric",
    },
}
