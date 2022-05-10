/*
Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// We need to define AT_PARALLEL_OPENMP (even if -fopenmp is
// not used) so that at::parallel_for is defined somewhere.
// This must be done before <ATen/Parallel.h> is included.
//
// Note that if AT_PARALLEL_OPENMP = 1 but compilation does not use
// -fopenmp, omp pragmas will be ignored. In that case, the code will
// be effectively sequential, and we don't have to worry about
// operations being atomic.
#if !(AT_PARALLEL_OPENMP)
#if !(AT_PARALLEL_NATIVE)
#if !(AT_PARALLEL_NATIVE_TBB)
#error No parallel backend specified
#endif
#endif
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// These are defines that help writing generic code for both GPU and CPU
#ifdef __CUDACC__
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#define MONAI_INLINE __forceinline__
#define MONAI_DEVICE __device__
#define MONAI_HOST __host__
#define MONAI_ATOMIC_ADD monai::gpuAtomicAdd
#define MONAI_NAMESPACE_DEVICE namespace cuda
namespace monai {
// atomicAdd API changed between pytorch 1.4 and 1.5.
template <typename scalar_t, typename offset_t>
static __forceinline__ __device__ void gpuAtomicAdd(scalar_t* ptr, offset_t offset, scalar_t value) {
#if MONAI_TORCH_VERSION >= 10500
  ::gpuAtomicAdd(ptr + offset, value);
#else
  ::atomicAdd(ptr + offset, value);
#endif
}
} // namespace monai
#else
#define MONAI_INLINE inline
#define MONAI_DEVICE
#define MONAI_HOST
#define MONAI_ATOMIC_ADD monai::cpuAtomicAdd
#define MONAI_NAMESPACE_DEVICE namespace cpu
namespace monai {
template <typename scalar_t, typename offset_t>
static inline void cpuAtomicAdd(scalar_t* ptr, offset_t offset, scalar_t value) {
#if AT_PARALLEL_OPENMP
#if _OPENMP
#pragma omp atomic
#endif
#endif
  ptr[offset] += value;
}
} // namespace monai
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <ATen/ATen.h>

namespace monai {

enum class BoundType : int64_t {
  Replicate, // Replicate last inbound value = clip coordinates
  DCT1, // Symmetric w.r.t. center of the last inbound voxel
  DCT2, // Symmetric w.r.t. edge of the last inbound voxel (=Neuman)
  DST1, // Asymmetric w.r.t. center of the last inbound voxel
  DST2, // Asymmetric w.r.t. edge of the last inbound voxel (=Dirichlet)
  DFT, // Circular / Wrap around the FOV
  Sliding, // For deformation-fields only: mixture of DCT2 and DST2
  Zero, // Zero outside of the FOV
  NoCheck // /!\ Checks disabled: assume coordinates are inbound
};

using BoundVectorRef = c10::ArrayRef<BoundType>;

enum class InterpolationType : int64_t {
  Nearest,
  Linear,
  Quadratic,
  Cubic,
  FourthOrder,
  FifthOrder,
  SixthOrder,
  SeventhOrder
};
using InterpolationVectorRef = c10::ArrayRef<InterpolationType>;

static MONAI_INLINE MONAI_HOST std::ostream& operator<<(std::ostream& os, const BoundType& bound) {
  switch (bound) {
    case BoundType::Replicate:
      return os << "Replicate";
    case BoundType::DCT1:
      return os << "DCT1";
    case BoundType::DCT2:
      return os << "DCT2";
    case BoundType::DST1:
      return os << "DST1";
    case BoundType::DST2:
      return os << "DST2";
    case BoundType::DFT:
      return os << "DFT";
    case BoundType::Zero:
      return os << "Zero";
    case BoundType::Sliding:
      return os << "Sliding";
    case BoundType::NoCheck:
      return os << "NoCheck";
  }
  return os << "Unknown bound";
}

static MONAI_INLINE MONAI_HOST std::ostream& operator<<(std::ostream& os, const InterpolationType& itp) {
  switch (itp) {
    case InterpolationType::Nearest:
      return os << "Nearest";
    case InterpolationType::Linear:
      return os << "Linear";
    case InterpolationType::Quadratic:
      return os << "Quadratic";
    case InterpolationType::Cubic:
      return os << "Cubic";
    case InterpolationType::FourthOrder:
      return os << "FourthOrder";
    case InterpolationType::FifthOrder:
      return os << "FifthOrder";
    case InterpolationType::SixthOrder:
      return os << "SixthOrder";
    case InterpolationType::SeventhOrder:
      return os << "SeventhOrder";
  }
  return os << "Unknown interpolation order";
}

} // namespace monai
