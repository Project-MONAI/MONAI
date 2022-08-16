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

# have to explicitly bring these in here to resolve circular import issues
from .aliases import alias, resolve_name
from .decorators import MethodReplacer, RestartGenerator
from .deprecate_utils import DeprecatedError, deprecated, deprecated_arg
from .dist import evenly_divisible_all_gather, get_dist_device, string_list_all_gather
from .enums import (
    Average,
    BlendMode,
    BoxModeName,
    ChannelMatching,
    ColorOrder,
    CommonKeys,
    DiceCEReduction,
    EngineStatsKeys,
    FastMRIKeys,
    ForwardMode,
    GanKeys,
    GridPatchSort,
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    InverseKeys,
    JITMetadataKeys,
    LossReduction,
    MetaKeys,
    Method,
    MetricReduction,
    NdimageMode,
    NumpyPadMode,
    PostFix,
    ProbMapKeys,
    PytorchPadMode,
    SkipMode,
    SpaceKeys,
    SplineMode,
    StrEnum,
    TraceKeys,
    TransformBackends,
    UpsampleMode,
    Weight,
    WSIPatchKeys,
)
from .jupyter_utils import StatusMembers, ThreadContainer
from .misc import (
    MAX_SEED,
    ImageMetaKey,
    check_parent_dir,
    copy_to_device,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    first,
    get_seed,
    has_option,
    is_module_ver_at_least,
    is_scalar,
    is_scalar_tensor,
    issequenceiterable,
    list_to_dict,
    progress_bar,
    sample_slices,
    save_obj,
    set_determinism,
    star_zip_with,
    zip_with,
)
from .module import (
    InvalidPyTorchVersionError,
    OptionalImportError,
    damerau_levenshtein_distance,
    exact_version,
    export,
    get_full_type_name,
    get_package_version,
    get_torch_version_tuple,
    instantiate,
    load_submodules,
    look_up_option,
    min_version,
    optional_import,
    pytorch_after,
    require_pkg,
    version_leq,
)
from .nvtx import Range
from .profiling import (
    PerfContext,
    ProfileHandler,
    WorkflowProfiler,
    select_transform_call,
    torch_profiler_full,
    torch_profiler_time_cpu_gpu,
    torch_profiler_time_end_to_end,
)
from .state_cacher import StateCacher
from .type_conversion import (
    convert_data_type,
    convert_to_cupy,
    convert_to_dst_type,
    convert_to_list,
    convert_to_numpy,
    convert_to_tensor,
    dtype_numpy_to_torch,
    dtype_torch_to_numpy,
    get_dtype,
    get_equivalent_dtype,
    get_numpy_dtype_from_string,
    get_torch_dtype_from_string,
)
