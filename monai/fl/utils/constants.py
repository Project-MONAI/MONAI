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


class ModelType(StrEnum):
    BEST_MODEL = "fl_best_model"
    FINAL_MODEL = "fl_final_model"


class ExtraItems(StrEnum):
    ABORT = "fl_abort"
    MODEL_TYPE = "fl_model_type"
    CLIENT_NAME = "fl_client_name"
    APP_ROOT = "fl_app_root"


class FlPhase(StrEnum):
    IDLE = "fl_idle"
    TRAIN = "fl_train"
    EVALUATE = "fl_evaluate"
    GET_WEIGHTS = "fl_get_weights"
    GET_DATA_STATS = "fl_get_data_stats"


class FlStatistics(StrEnum):
    NUM_EXECUTED_ITERATIONS = "num_executed_iterations"
    STATISTICS = "statistics"
    HIST_BINS = "hist_bins"
    HIST_RANGE = "hist_range"
    DATA_STATS = "data_stats"
    DATA_COUNT = "data_count"
    FAIL_COUNT = "fail_count"
    TOTAL_DATA = "total_data"
    FEATURE_NAMES = "feature_names"


class RequiredBundleKeys(StrEnum):
    BUNDLE_ROOT = "bundle_root"


class BundleKeys(StrEnum):
    TRAINER = "train#trainer"
    EVALUATOR = "validate#evaluator"
    TRAIN_TRAINER_MAX_EPOCHS = "train#trainer#max_epochs"
    VALIDATE_HANDLERS = "validate#handlers"
    DATASET_DIR = "dataset_dir"
    TRAIN_DATA = "train#dataset#data"
    VALID_DATA = "validate#dataset#data"


class FiltersType(StrEnum):
    PRE_FILTERS = "pre_filters"
    POST_WEIGHT_FILTERS = "post_weight_filters"
    POST_EVALUATE_FILTERS = "post_evaluate_filters"
    POST_STATISTICS_FILTERS = "post_statistics_filters"
