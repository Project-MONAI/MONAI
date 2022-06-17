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

// adapted from https://github.com/balbasty/nitorch

#pragma once

// This file contains static functions for handling (0-7 order)
// interpolation weights.
// It also defines an enumerated types that encodes each boundary type.
// The entry points are:
// . monai::interpolation::weight     -> node weight based on distance
// . monai::interpolation::fastweight -> same, assuming x lies in support
// . monai::interpolation::grad       -> weight derivative // oriented distance
// . monai::interpolation::fastgrad   -> same, assuming x lies in support
// . monai::interpolation::hess       -> weight 2nd derivative // oriented distance
// . monai::interpolation::fasthess   -> same, assuming x lies in support
// . monai::interpolation::bounds     -> min/max nodes

// NOTE:
// 1st derivatives used to be implemented with a recursive call, e.g.:
// scalar_t grad2(scalar_t x) {
//   if (x < 0) return -grad2(-x);
//   ...
// }
// However, this prevents nvcc to statically determine the stack size
// and leads to memory errors (because the allocated stack is too small).
// I now use a slightly less compact implementation that gets rid of
// recursive calls.

// TODO:
// . second order derivatives [5/6/7]
// ? other types of basis functions (gauss, sinc)

#include "utils/resample_utils.h"

namespace monai {

namespace _interpolation {

// --- order 0 -------------------------------------------------------

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t weight0(scalar_t x) {
  x = std::fabs(x);
  return x < 0.5 ? static_cast<scalar_t>(1) : static_cast<scalar_t>(0);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastweight0(scalar_t x) {
  x = std::fabs(x);
  return static_cast<scalar_t>(1);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t grad0(scalar_t x) {
  return static_cast<scalar_t>(0);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastgrad0(scalar_t x) {
  return static_cast<scalar_t>(0);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t hess0(scalar_t x) {
  return static_cast<scalar_t>(0);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fasthess0(scalar_t x) {
  return static_cast<scalar_t>(0);
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void bounds0(scalar_t x, offset_t& low, offset_t& upp) {
  low = static_cast<offset_t>(std::round(x));
  upp = low;
}

// --- order 1 -------------------------------------------------------

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t weight1(scalar_t x) {
  x = std::fabs(x);
  return x < 1 ? static_cast<scalar_t>(1) - x : static_cast<scalar_t>(0);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastweight1(scalar_t x) {
  return static_cast<scalar_t>(1) - std::fabs(x);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t grad1(scalar_t x) {
  if (std::fabs(x) >= 1)
    return static_cast<scalar_t>(0);
  return fastgrad1(x);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastgrad1(scalar_t x) {
  return x < static_cast<scalar_t>(0) ? static_cast<scalar_t>(1) : static_cast<scalar_t>(-1);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t hess1(scalar_t x) {
  return static_cast<scalar_t>(0);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fasthess1(scalar_t x) {
  return static_cast<scalar_t>(0);
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void bounds1(scalar_t x, offset_t& low, offset_t& upp) {
  low = static_cast<offset_t>(std::floor(x));
  upp = low + 1;
}

// --- order 2 -------------------------------------------------------

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t weight2(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    return 0.75 - x * x;
  } else if (x < 1.5) {
    x = 1.5 - x;
    return 0.5 * x * x;
  } else {
    return static_cast<scalar_t>(0);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastweight2(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    return 0.75 - x * x;
  } else {
    x = 1.5 - x;
    return 0.5 * x * x;
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t grad2(scalar_t x) {
  bool neg = x < 0;
  if (x < 0.5) {
    x = -2. * x;
  } else if (x < 1.5) {
    x = x - 1.5;
  } else {
    return static_cast<scalar_t>(0);
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastgrad2(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < 0.5) {
    x = -2. * x;
  } else {
    x = x - 1.5;
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t hess2(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    return static_cast<scalar_t>(-2.);
  } else if (x < 1.5) {
    return static_cast<scalar_t>(1.);
  } else {
    return static_cast<scalar_t>(0);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fasthess2(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    return static_cast<scalar_t>(-2.);
  } else {
    return static_cast<scalar_t>(1.);
  }
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void bounds2(scalar_t x, offset_t& low, offset_t& upp) {
  low = static_cast<offset_t>(std::floor(x - .5));
  upp = low + 2;
}

// --- order 3 -------------------------------------------------------

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t weight3(scalar_t x) {
  x = std::fabs(x);
  if (x < 1.) {
    return (x * x * (x - 2.) * 3. + 4.) / 6.;
  } else if (x < 2.) {
    x = 2. - x;
    return (x * x * x) / 6.;
  } else {
    return static_cast<scalar_t>(0);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastweight3(scalar_t x) {
  x = std::fabs(x);
  if (x < 1.) {
    return (x * x * (x - 2.) * 3. + 4.) / 6.;
  } else {
    x = 2. - x;
    return (x * x * x) / 6.;
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t grad3(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < 1.) {
    x = x * (x * 1.5 - 2.);
  } else if (x < 2.) {
    x = 2. - x;
    x = -(x * x) * 0.5;
  } else {
    return static_cast<scalar_t>(0);
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastgrad3(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < 1.) {
    x = x * (x * 1.5 - 2.);
  } else {
    x = 2. - x;
    x = -(x * x) * 0.5;
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t hess3(scalar_t x) {
  x = std::fabs(x);
  if (x < 1.) {
    return x * 3. - 2.;
  } else if (x < 2.) {
    return 2. - x;
  } else {
    return static_cast<scalar_t>(0);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fasthess3(scalar_t x) {
  x = std::fabs(x);
  if (x < 1.) {
    return x * 3. - 2.;
  } else {
    return 2. - x;
  }
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void bounds3(scalar_t x, offset_t& low, offset_t& upp) {
  low = static_cast<offset_t>(std::floor(x - 1.));
  upp = low + 3;
}

// --- order 4 -------------------------------------------------------

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t weight4(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    x *= x;
    return x * (x * 0.25 - 0.625) + 115. / 192.;
  } else if (x < 1.5) {
    return x * (x * (x * (5. - x) / 6. - 1.25) + 5. / 24.) + 55. / 96.;
  } else if (x < 2.5) {
    x -= 2.5;
    x *= x;
    return (x * x) / 24.;
  } else {
    return static_cast<scalar_t>(0);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastweight4(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    x *= x;
    return x * (x * 0.25 - 0.625) + 115. / 192.;
  } else if (x < 1.5) {
    return x * (x * (x * (5. - x) / 6. - 1.25) + 5. / 24.) + 55. / 96.;
  } else {
    x -= 2.5;
    x *= x;
    return (x * x) / 24.;
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t grad4(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < 0.5) {
    x = x * (x * x - 1.25);
  } else if (x < 1.5) {
    x = x * (x * (x * (-2. / 3.) + 2.5) - 2.5) + 5. / 24.;
  } else if (x < 2.5) {
    x = x * 2. - 5.;
    x = (x * x * x) / 48.;
  } else {
    return static_cast<scalar_t>(0);
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastgrad4(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < 0.5) {
    x = x * (x * x - 1.25);
  } else if (x < 1.5) {
    x = x * (x * (x * (-2. / 3.) + 2.5) - 2.5) + 5. / 24.;
  } else {
    x = x * 2. - 5.;
    x = (x * x * x) / 48.;
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t hess4(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    return (x * x) * 3. - 1.25;
  } else if (x < 1.5) {
    return x * (x * (-2.) + 5.) - 2.5;
  } else if (x < 2.5) {
    x = x * 2. - 5.;
    return (x * x) / 8.;
  } else {
    return static_cast<scalar_t>(0);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fasthess4(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    return (x * x) * 3. - 1.25;
  } else if (x < 1.5) {
    return x * (x * (-2.) + 5.) - 2.5;
  } else {
    x = x * 2. - 5.;
    return (x * x) / 8.;
  }
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void bounds4(scalar_t x, offset_t& low, offset_t& upp) {
  low = static_cast<offset_t>(std::floor(x - 1.5));
  upp = low + 4;
}

// --- order 5 -------------------------------------------------------

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t weight5(scalar_t x) {
  x = std::fabs(x);
  if (x < 1.) {
    scalar_t f = x * x;
    return f * (f * (0.25 - x * (1. / 12.)) - 0.5) + 0.55;
  } else if (x < 2.) {
    return x * (x * (x * (x * (x * (1. / 24.) - 0.375) + 1.25) - 1.75) + 0.625) + 0.425;
  } else if (x < 3.) {
    scalar_t f = 3. - x;
    x = f * f;
    return f * x * x * (1. / 120.);
  } else
    return static_cast<scalar_t>(0);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastweight5(scalar_t x) {
  x = std::fabs(x);
  if (x < 1.) {
    scalar_t f = x * x;
    return f * (f * (0.25 - x * (1. / 12.)) - 0.5) + 0.55;
  } else if (x < 2.) {
    return x * (x * (x * (x * (x * (1. / 24.) - 0.375) + 1.25) - 1.75) + 0.625) + 0.425;
  } else {
    scalar_t f = 3. - x;
    x = f * f;
    return f * x * x * (1. / 120.);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t grad5(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < 1.) {
    x = x * (x * (x * (x * (-5. / 12.) + 1.)) - 1.);
  } else if (x < 2.) {
    x = x * (x * (x * (x * (5. / 24.) - 1.5) + 3.75) - 3.5) + 0.625;
  } else if (x < 3.) {
    x -= 3.;
    x *= x;
    x = -(x * x) / 24.;
  } else {
    return static_cast<scalar_t>(0);
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastgrad5(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < 1.) {
    x = x * (x * (x * (x * (-5. / 12.) + 1.)) - 1.);
  } else if (x < 2.) {
    x = x * (x * (x * (x * (5. / 24.) - 1.5) + 3.75) - 3.5) + 0.625;
  } else {
    x -= 3.;
    x *= x;
    x = -(x * x) / 24.;
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void bounds5(scalar_t x, offset_t& low, offset_t& upp) {
  low = static_cast<offset_t>(std::floor(x - 2.));
  upp = low + 5;
}

// --- order 6 -------------------------------------------------------

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t weight6(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    x *= x;
    return x * (x * (7. / 48. - x * (1. / 36.)) - 77. / 192.) + 5887. / 11520.0;
  } else if (x < 1.5) {
    return x * (x * (x * (x * (x * (x * (1. / 48.) - 7. / 48.) + 0.328125) - 35. / 288.) - 91. / 256.) - 7. / 768.) +
        7861. / 15360.0;
  } else if (x < 2.5) {
    return x * (x * (x * (x * (x * (7. / 60. - x * (1. / 120.)) - 0.65625) + 133. / 72.) - 2.5703125) + 1267. / 960.) +
        1379. / 7680.0;
  } else if (x < 3.5) {
    x -= 3.5;
    x *= x * x;
    return x * x * (1. / 720.);
  } else
    return static_cast<scalar_t>(0);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastweight6(scalar_t x) {
  x = std::fabs(x);
  if (x < 0.5) {
    x *= x;
    return x * (x * (7. / 48. - x * (1. / 36.)) - 77. / 192.) + 5887. / 11520.0;
  } else if (x < 1.5) {
    return x * (x * (x * (x * (x * (x * (1. / 48.) - 7. / 48.) + 0.328125) - 35. / 288.) - 91. / 256.) - 7. / 768.) +
        7861. / 15360.0;
  } else if (x < 2.5) {
    return x * (x * (x * (x * (x * (7. / 60. - x * (1. / 120.)) - 0.65625) + 133. / 72.) - 2.5703125) + 1267. / 960.) +
        1379. / 7680.0;
  } else {
    x -= 3.5;
    x *= x * x;
    return x * x * (1. / 720.);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t grad6(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < .5) {
    scalar_t x2 = x * x;
    x = x * (x2 * (7. / 12.) - (x2 * x2) / 6. - 77. / 96.);
  } else if (x < 1.5) {
    x = x * (x * (x * (x * (x * 0.125 - 35. / 48.) + 1.3125) - 35. / 96.) - 0.7109375) - 7.0 / 768.0;
  } else if (x < 2.5) {
    x = x * (x * (x * (x * (x * (-1. / 20.) + 7. / 12.) - 2.625) + 133. / 24.) - 5.140625) + 1267. / 960.;
  } else if (x < 3.5) {
    x *= 2.;
    x -= 7.;
    scalar_t x2 = x * x;
    x = (x2 * x2 * x) / 3840.;
  } else {
    return static_cast<scalar_t>(0);
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastgrad6(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < .5) {
    scalar_t x2 = x * x;
    x = x * (x2 * (7. / 12.) - (x2 * x2) / 6. - 77. / 96.);
  } else if (x < 1.5) {
    x = x * (x * (x * (x * (x * 0.125 - 35. / 48.) + 1.3125) - 35. / 96.) - 0.7109375) - 7.0 / 768.0;
  } else if (x < 2.5) {
    x = x * (x * (x * (x * (x * (-1. / 20.) + 7. / 12.) - 2.625) + 133. / 24.) - 5.140625) + 1267. / 960.;
  } else {
    x *= 2.;
    x -= 7.;
    scalar_t x2 = x * x;
    x = (x2 * x2 * x) / 3840.;
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void bounds6(scalar_t x, offset_t& low, offset_t& upp) {
  low = static_cast<offset_t>(std::floor(x - 2.5));
  upp = low + 6;
}

// --- order 7 -------------------------------------------------------

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t weight7(scalar_t x) {
  x = std::fabs(x);
  if (x < 1.) {
    scalar_t f = x * x;
    return f * (f * (f * (x * (1. / 144.) - 1. / 36.) + 1. / 9.) - 1. / 3.) + 151. / 315.0;
  } else if (x < 2.) {
    return x * (x * (x * (x * (x * (x * (0.05 - x * (1. / 240.)) - 7. / 30.) + 0.5) - 7. / 18.) - 0.1) - 7. / 90.) +
        103. / 210.0;
  } else if (x < 3.) {
    return x *
        (x * (x * (x * (x * (x * (x * (1. / 720.) - 1. / 36.) + 7. / 30.) - 19. / 18.) + 49. / 18.) - 23. / 6.) +
         217. / 90.) -
        139. / 630.0;
  } else if (x < 4.) {
    scalar_t f = 4. - x;
    x = f * f * f;
    return (x * x * f) / 5040.;
  } else
    return static_cast<scalar_t>(0);
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastweight7(scalar_t x) {
  x = std::fabs(x);
  if (x < 1.) {
    scalar_t f = x * x;
    return f * (f * (f * (x * (1. / 144.) - 1. / 36.) + 1. / 9.) - 1. / 3.) + 151. / 315.0;
  } else if (x < 2.) {
    return x * (x * (x * (x * (x * (x * (0.05 - x * (1. / 240.)) - 7. / 30.) + 0.5) - 7. / 18.) - 0.1) - 7. / 90.) +
        103. / 210.0;
  } else if (x < 3.) {
    return x *
        (x * (x * (x * (x * (x * (x * (1. / 720.) - 1. / 36.) + 7. / 30.) - 19. / 18.) + 49. / 18.) - 23. / 6.) +
         217. / 90.) -
        139. / 630.0;
  } else {
    scalar_t f = 4. - x;
    x = f * f * f;
    return (x * x * f) / 5040.;
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t grad7(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < 1.) {
    scalar_t x2 = x * x;
    x = x * (x2 * (x2 * (x * (7. / 144.) - 1. / 6.) + 4. / 9.) - 2. / 3.);
  } else if (x < 2.) {
    x = x * (x * (x * (x * (x * (x * (-7. / 240.) + 3. / 10.) - 7. / 6.) + 2.) - 7. / 6.) - 1. / 5.) - 7. / 90.;
  } else if (x < 3.) {
    x = x * (x * (x * (x * (x * (x * (7. / 720.) - 1. / 6.) + 7. / 6.) - 38. / 9.) + 49. / 6.) - 23. / 3.) + 217. / 90.;
  } else if (x < 4.) {
    x -= 4;
    x *= x * x;
    x *= x;
    x = -x / 720.;
  } else {
    return static_cast<scalar_t>(0);
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastgrad7(scalar_t x) {
  bool neg = x < 0;
  if (neg)
    x = -x;
  if (x < 1.) {
    scalar_t x2 = x * x;
    x = x * (x2 * (x2 * (x * (7. / 144.) - 1. / 6.) + 4. / 9.) - 2. / 3.);
  } else if (x < 2.) {
    x = x * (x * (x * (x * (x * (x * (-7. / 240.) + 3. / 10.) - 7. / 6.) + 2.) - 7. / 6.) - 1. / 5.) - 7. / 90.;
  } else if (x < 3.) {
    x = x * (x * (x * (x * (x * (x * (7. / 720.) - 1. / 6.) + 7. / 6.) - 38. / 9.) + 49. / 6.) - 23. / 3.) + 217. / 90.;
  } else {
    x -= 4;
    x *= x * x;
    x *= x;
    x = -x / 720.;
  }
  if (neg)
    x = -x;
  return x;
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void bounds7(scalar_t x, offset_t& low, offset_t& upp) {
  low = static_cast<offset_t>(std::floor(x - 3.));
  upp = low + 7;
}

} // namespace _interpolation

namespace interpolation {

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t weight(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:
      return _interpolation::weight0(x);
    case InterpolationType::Linear:
      return _interpolation::weight1(x);
    case InterpolationType::Quadratic:
      return _interpolation::weight2(x);
    case InterpolationType::Cubic:
      return _interpolation::weight3(x);
    case InterpolationType::FourthOrder:
      return _interpolation::weight4(x);
    case InterpolationType::FifthOrder:
      return _interpolation::weight5(x);
    case InterpolationType::SixthOrder:
      return _interpolation::weight6(x);
    case InterpolationType::SeventhOrder:
      return _interpolation::weight7(x);
    default:
      return _interpolation::weight1(x);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastweight(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:
      return _interpolation::fastweight0(x);
    case InterpolationType::Linear:
      return _interpolation::fastweight1(x);
    case InterpolationType::Quadratic:
      return _interpolation::fastweight2(x);
    case InterpolationType::Cubic:
      return _interpolation::fastweight3(x);
    case InterpolationType::FourthOrder:
      return _interpolation::fastweight4(x);
    case InterpolationType::FifthOrder:
      return _interpolation::fastweight5(x);
    case InterpolationType::SixthOrder:
      return _interpolation::fastweight6(x);
    case InterpolationType::SeventhOrder:
      return _interpolation::fastweight7(x);
    default:
      return _interpolation::fastweight1(x);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t grad(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:
      return _interpolation::grad0(x);
    case InterpolationType::Linear:
      return _interpolation::grad1(x);
    case InterpolationType::Quadratic:
      return _interpolation::grad2(x);
    case InterpolationType::Cubic:
      return _interpolation::grad3(x);
    case InterpolationType::FourthOrder:
      return _interpolation::grad4(x);
    case InterpolationType::FifthOrder:
      return _interpolation::grad5(x);
    case InterpolationType::SixthOrder:
      return _interpolation::grad6(x);
    case InterpolationType::SeventhOrder:
      return _interpolation::grad7(x);
    default:
      return _interpolation::grad1(x);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fastgrad(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:
      return _interpolation::fastgrad0(x);
    case InterpolationType::Linear:
      return _interpolation::fastgrad1(x);
    case InterpolationType::Quadratic:
      return _interpolation::fastgrad2(x);
    case InterpolationType::Cubic:
      return _interpolation::fastgrad3(x);
    case InterpolationType::FourthOrder:
      return _interpolation::fastgrad4(x);
    case InterpolationType::FifthOrder:
      return _interpolation::fastgrad5(x);
    case InterpolationType::SixthOrder:
      return _interpolation::fastgrad6(x);
    case InterpolationType::SeventhOrder:
      return _interpolation::fastgrad7(x);
    default:
      return _interpolation::fastgrad1(x);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t hess(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:
      return _interpolation::hess0(x);
    case InterpolationType::Linear:
      return _interpolation::hess1(x);
    case InterpolationType::Quadratic:
      return _interpolation::hess2(x);
    case InterpolationType::Cubic:
      return _interpolation::hess3(x);
    case InterpolationType::FourthOrder:
      return _interpolation::hess4(x);
    case InterpolationType::FifthOrder:
      return _interpolation::hess0(x); // notimplemented
    case InterpolationType::SixthOrder:
      return _interpolation::hess0(x); // notimplemented
    case InterpolationType::SeventhOrder:
      return _interpolation::hess0(x); // notimplemented
    default:
      return _interpolation::grad1(x);
  }
}

template <typename scalar_t>
static MONAI_INLINE MONAI_DEVICE scalar_t fasthess(InterpolationType interpolation_type, scalar_t x) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:
      return _interpolation::fasthess0(x);
    case InterpolationType::Linear:
      return _interpolation::fasthess1(x);
    case InterpolationType::Quadratic:
      return _interpolation::fasthess2(x);
    case InterpolationType::Cubic:
      return _interpolation::fasthess3(x);
    case InterpolationType::FourthOrder:
      return _interpolation::fasthess4(x);
    case InterpolationType::FifthOrder:
      return _interpolation::fasthess0(x); // notimplemented
    case InterpolationType::SixthOrder:
      return _interpolation::fasthess0(x); // notimplemented
    case InterpolationType::SeventhOrder:
      return _interpolation::fasthess0(x); // notimplemented
    default:
      return _interpolation::fasthess1(x);
  }
}

template <typename scalar_t, typename offset_t>
static MONAI_INLINE MONAI_DEVICE void bounds(
    InterpolationType interpolation_type,
    scalar_t x,
    offset_t& low,
    offset_t& upp) {
  switch (interpolation_type) {
    case InterpolationType::Nearest:
      return _interpolation::bounds0(x, low, upp);
    case InterpolationType::Linear:
      return _interpolation::bounds1(x, low, upp);
    case InterpolationType::Quadratic:
      return _interpolation::bounds2(x, low, upp);
    case InterpolationType::Cubic:
      return _interpolation::bounds3(x, low, upp);
    case InterpolationType::FourthOrder:
      return _interpolation::bounds4(x, low, upp);
    case InterpolationType::FifthOrder:
      return _interpolation::bounds5(x, low, upp);
    case InterpolationType::SixthOrder:
      return _interpolation::bounds6(x, low, upp);
    case InterpolationType::SeventhOrder:
      return _interpolation::bounds7(x, low, upp);
    default:
      return _interpolation::bounds1(x, low, upp);
  }
}

} // namespace interpolation

} // namespace monai
