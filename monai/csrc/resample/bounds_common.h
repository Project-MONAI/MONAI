/*
Copyright 2020 MONAI Consortium
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

// adapted from https://github.com/balbasty/nitorch

#pragma once

// This file contains static functions for handling out-of-bound indices.
// They implement typical boundary conditions (those of standard discrete
// transforms) + a few other cases (replicated border, zeros, sliding)
// It also defines an enumerated types that encodes each boundary type.
// The entry points are:
// . monai::bound::index -> wrap out-of-bound indices
// . monai::bound::sign  -> optional out-of-bound sign change (sine transforms)
// . monai::BoundType    -> enumerated boundary type
//
// Everything in this file should have internal linkage (static) except
// the BoundType/BoundVectorRef types.

#include "utils/resample_utils.h"

namespace monai {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             INDEXING
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace _index {

template <typename size_t>
static MONAI_INLINE MONAI_DEVICE size_t inbounds(size_t coord, size_t size) {
  return coord;
}

// Boundary condition of a DCT-I (periodicity: (n-1)*2)
// Indices are reflected about the centre of the border elements:
//    -1 --> 1
//     n --> n-2
template <typename size_t>
static MONAI_INLINE MONAI_DEVICE size_t reflect1c(size_t coord, size_t size) {
  if (size == 1)
    return 0;
  size_t size_twice = (size - 1) * 2;
  coord = coord < 0 ? -coord : coord;
  coord = coord % size_twice;
  coord = coord >= size ? size_twice - coord : coord;
  return coord;
}

// Boundary condition of a DST-I (periodicity: (n+1)*2)
// Indices are reflected about the centre of the first out-of-bound
// element:
//    -1 --> undefined [0]
//    -2 --> 0
//     n --> undefined [n-1]
//   n+1 --> n-1
template <typename size_t>
static MONAI_INLINE MONAI_DEVICE size_t reflect1s(size_t coord, size_t size) {
  if (size == 1)
    return 0;
  size_t size_twice = (size + 1) * 2;
  coord = coord == -1 ? 0 : coord < 0 ? -coord - 2 : coord;
  coord = coord % size_twice;
  coord = coord == size ? size - 1 : coord > size ? size_twice - coord - 2 : coord;
  return coord;
}

// Boundary condition of a DCT/DST-II (periodicity: n*2)
// Indices are reflected about the edge of the border elements:
//    -1 --> 0
//     n --> n-1
template <typename size_t>
static MONAI_INLINE MONAI_DEVICE size_t reflect2(size_t coord, size_t size) {
  size_t size_twice = size * 2;
  coord = coord < 0 ? size_twice - ((-coord - 1) % size_twice) - 1 : coord % size_twice;
  coord = coord >= size ? size_twice - coord - 1 : coord;
  return coord;
}

// Boundary condition of a DFT (periodicity: n)
// Indices wrap about the edges:
//    -1 --> n-1
//     n --> 0
template <typename size_t>
static MONAI_INLINE MONAI_DEVICE size_t circular(size_t coord, size_t size) {
  coord = coord < 0 ? (size + coord % size) % size : coord % size;
  return coord;
}

// Replicate edge values:
//    -1 --> 0
//    -2 --> 0
//     n --> n-1
//   n+1 --> n-1
template <typename size_t>
static MONAI_INLINE MONAI_DEVICE size_t replicate(size_t coord, size_t size) {
  coord = coord <= 0 ? 0 : coord >= size ? size - 1 : coord;
  return coord;
}

} // namespace _index

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          SIGN MODIFICATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace _sign {

template <typename size_t>
static MONAI_INLINE MONAI_DEVICE int8_t inbounds(size_t coord, size_t size) {
  return coord < 0 || coord >= size ? 0 : 1;
}

// Boundary condition of a DCT/DFT
// No sign modification based on coordinates
template <typename size_t>
static MONAI_INLINE MONAI_DEVICE int8_t constant(size_t coord, size_t size) {
  return static_cast<int8_t>(1);
}

// Boundary condition of a DST-I
// Periodic sign change based on coordinates
template <typename size_t>
static MONAI_INLINE MONAI_DEVICE int8_t periodic1(size_t coord, size_t size) {
  if (size == 1)
    return 1;
  size_t size_twice = (size + 1) * 2;
  coord = coord < 0 ? size - coord - 1 : coord;
  coord = coord % size_twice;
  if (coord % (size + 1) == size)
    return static_cast<int8_t>(0);
  else if ((coord / (size + 1)) % 2)
    return static_cast<int8_t>(-1);
  else
    return static_cast<int8_t>(1);
}

// Boundary condition of a DST-II
// Periodic sign change based on coordinates
template <typename size_t>
static MONAI_INLINE MONAI_DEVICE int8_t periodic2(size_t coord, size_t size) {
  coord = (coord < 0 ? size - coord - 1 : coord);
  return static_cast<int8_t>((coord / size) % 2 ? -1 : 1);
}

} // namespace _sign

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                BOUND
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check if coordinates within bounds
template <typename size_t>
static MONAI_INLINE MONAI_DEVICE bool inbounds(size_t coord, size_t size) {
  return coord >= 0 && coord < size;
}

template <typename scalar_t, typename size_t>
static MONAI_INLINE MONAI_DEVICE bool inbounds(scalar_t coord, size_t size, scalar_t tol) {
  return coord >= -tol && coord < (scalar_t)(size - 1) + tol;
}

namespace bound {

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE scalar_t
get(const scalar_t* ptr, offset_t offset, int8_t sign = static_cast<int8_t>(1)) {
  if (sign == -1)
    return -ptr[offset];
  else if (sign)
    return ptr[offset];
  else
    return static_cast<scalar_t>(0);
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void add(
    scalar_t* ptr,
    offset_t offset,
    scalar_t val,
    int8_t sign = static_cast<int8_t>(1)) {
  if (sign == -1)
    MONAI_ATOMIC_ADD(ptr, offset, -val);
  else if (sign)
    MONAI_ATOMIC_ADD(ptr, offset, val);
}

template <typename size_t>
static MONAI_INLINE MONAI_DEVICE int64_t index(BoundType bound_type, size_t coord, size_t size) {
  switch (bound_type) {
    case BoundType::Replicate:
      return _index::replicate(coord, size);
    case BoundType::DCT1:
      return _index::reflect1c(coord, size);
    case BoundType::DCT2:
      return _index::reflect2(coord, size);
    case BoundType::DST1:
      return _index::reflect1s(coord, size);
    case BoundType::DST2:
      return _index::reflect2(coord, size);
    case BoundType::DFT:
      return _index::circular(coord, size);
    case BoundType::Zero:
      return _index::inbounds(coord, size);
    default:
      return _index::inbounds(coord, size);
  }
}

template <typename size_t>
static MONAI_INLINE MONAI_DEVICE int8_t sign(BoundType bound_type, size_t coord, size_t size) {
  switch (bound_type) {
    case BoundType::Replicate:
      return _sign::constant(coord, size);
    case BoundType::DCT1:
      return _sign::constant(coord, size);
    case BoundType::DCT2:
      return _sign::constant(coord, size);
    case BoundType::DST1:
      return _sign::periodic1(coord, size);
    case BoundType::DST2:
      return _sign::periodic2(coord, size);
    case BoundType::DFT:
      return _sign::constant(coord, size);
    case BoundType::Zero:
      return _sign::inbounds(coord, size);
    default:
      return _sign::inbounds(coord, size);
  }
}

} // namespace bound
} // namespace monai
