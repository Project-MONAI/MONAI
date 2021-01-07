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

# have to explicitly bring these in here to resolve circular import issues
from .aliases import alias, resolve_name
from .decorators import MethodReplacer, RestartGenerator
from .enums import (
    Activation,
    Average,
    BlendMode,
    ChannelMatching,
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    LossReduction,
    Method,
    MetricReduction,
    Normalization,
    NumpyPadMode,
    PytorchPadMode,
    SkipMode,
    UpsampleMode,
    Weight,
)
from .misc import (
    MAX_SEED,
    dtype_numpy_to_torch,
    dtype_torch_to_numpy,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    first,
    get_seed,
    is_scalar,
    is_scalar_tensor,
    issequenceiterable,
    list_to_dict,
    progress_bar,
    set_determinism,
    star_zip_with,
    zip_with,
)
from .module import (
    PT_BEFORE_1_7,
    InvalidPyTorchVersionError,
    OptionalImportError,
    exact_version,
    export,
    get_full_type_name,
    get_package_version,
    get_torch_version_tuple,
    has_option,
    load_submodules,
    min_version,
    optional_import,
)
from .profiling import PerfContext, torch_profiler_full, torch_profiler_time_cpu_gpu, torch_profiler_time_end_to_end
