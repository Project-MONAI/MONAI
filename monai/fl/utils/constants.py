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


class WeightType:
    WEIGHTS = "fl_weights_full"
    WEIGHT_DIFF = "fl_weight_diff"


class ExtraItems:
    ABORT = "fl_abort"
    MODEL_NAME = "fl_model_name"
    CLIENT_NAME = "fl_client_name"


class FlPhase:
    IDLE = "fl_idle"
    TRAIN = "fl_train"
    EVALUATE = "fl_evaluate"
    GET_WEIGHTS = "fl_get_weights"


class FlStatistics:
    # TODO: currently uses hardcoded strings from. Might move to MONAI core?
    #  https://github.com/Project-MONAI/MONAI/blob/df4a7d72e1d231b898f88d92cf981721c49ceaeb/monai/engines/trainer.py#L60
    TOTAL_EPOCHS = "total_epochs"
    TOTAL_ITERATIONS = "total_iterations"


class BundleConst:
    KEY_TRAINER = "train#trainer"
    KEY_EVALUATOR = "validate#evaluator"


class FiltersType:
    PRE_FILTERS = "pre_filters"
    POST_WEIGHT_FILTERS = "post_weight_filters"
    POST_EVALUATE_FILTERS = "post_evaluate_filters"
