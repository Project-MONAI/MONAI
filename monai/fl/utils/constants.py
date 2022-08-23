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

from monai.utils.enums import StrEnum


class WeightType(StrEnum):
    WEIGHTS = "fl_weights_full"
    WEIGHT_DIFF = "fl_weight_diff"


class ExtraItems(StrEnum):
    ABORT = "fl_abort"
    MODEL_NAME = "fl_model_name"
    CLIENT_NAME = "fl_client_name"
    APP_ROOT = "fl_app_root"


class FlPhase(StrEnum):
    IDLE = "fl_idle"
    TRAIN = "fl_train"
    EVALUATE = "fl_evaluate"
    GET_WEIGHTS = "fl_get_weights"


class FlStatistics(StrEnum):
    NUM_EXECUTED_ITERATIONS = "num_executed_iterations"


class RequiredBundleKeys(StrEnum):
    BUNDLE_ROOT = "bundle_root"


class BundleKeys(StrEnum):
    TRAINER = "train#trainer"
    EVALUATOR = "validate#evaluator"
    TRAIN_TRAINER_MAX_EPOCHS = "train#trainer#max_epochs"
    VALIDATE_HANDLERS = "validate#handlers"


class FiltersType(StrEnum):
    PRE_FILTERS = "pre_filters"
    POST_WEIGHT_FILTERS = "post_weight_filters"
    POST_EVALUATE_FILTERS = "post_evaluate_filters"
