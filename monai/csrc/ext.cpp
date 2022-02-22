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

#include <torch/extension.h>

#include "filtering/filtering.h"
#include "lltm/lltm.h"
#include "resample/pushpull.h"
#include "utils/resample_utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // filtering
  m.def("bilateral_filter", &BilateralFilter, "Bilateral Filter");
  m.def("phl_filter", &PermutohedralFilter, "Permutohedral Filter");

  // lltm
  m.def("lltm_forward", &lltm_forward, "LLTM forward");
  m.def("lltm_backward", &lltm_backward, "LLTM backward");

  // resample bound mode
  py::enum_<monai::BoundType>(m, "BoundType")
      .value("replicate", monai::BoundType::Replicate, "a a a | a b c d | d d d")
      .value("nearest", monai::BoundType::Replicate, "a a a | a b c d | d d d")
      .value("border", monai::BoundType::Replicate, "a a a | a b c d | d d d")
      .value("dct1", monai::BoundType::DCT1, "d c b | a b c d | c b a")
      .value("mirror", monai::BoundType::DCT1, "d c b | a b c d | c b a")
      .value("dct2", monai::BoundType::DCT2, "c b a | a b c d | d c b")
      .value("reflect", monai::BoundType::DCT2, "c b a | a b c d | d c b")
      .value("dst1", monai::BoundType::DST1, "-b -a 0 | a b c d | 0 -d -c")
      .value("antimirror", monai::BoundType::DST1, "-b -a 0 | a b c d | 0 -d -c")
      .value("dst2", monai::BoundType::DST2, "-c -b -a | a b c d | -d -c -b")
      .value("antireflect", monai::BoundType::DST2, "-c -b -a | a b c d | -d -c -b")
      .value("dft", monai::BoundType::DFT, "b c d | a b c d | a b c")
      .value("wrap", monai::BoundType::DFT, "b c d | a b c d | a b c")
      //   .value("sliding", monai::BoundType::Sliding)
      .value("zero", monai::BoundType::Zero, "0 0 0 | a b c d | 0 0 0")
      .value("zeros", monai::BoundType::Zero, "0 0 0 | a b c d | 0 0 0")
      .export_values();

  // resample interpolation mode
  py::enum_<monai::InterpolationType>(m, "InterpolationType")
      .value("nearest", monai::InterpolationType::Nearest)
      .value("linear", monai::InterpolationType::Linear)
      .value("quadratic", monai::InterpolationType::Quadratic)
      .value("cubic", monai::InterpolationType::Cubic)
      .value("fourth", monai::InterpolationType::FourthOrder)
      .value("fifth", monai::InterpolationType::FifthOrder)
      .value("sixth", monai::InterpolationType::SixthOrder)
      .value("seventh", monai::InterpolationType::SeventhOrder)
      .export_values();

  // resample
  m.def("grid_pull", &monai::grid_pull, "GridPull");
  m.def("grid_pull_backward", &monai::grid_pull_backward, "GridPull backward");
  m.def("grid_push", &monai::grid_push, "GridPush");
  m.def("grid_push_backward", &monai::grid_push_backward, "GridPush backward");
  m.def("grid_count", &monai::grid_count, "GridCount");
  m.def("grid_count_backward", &monai::grid_count_backward, "GridCount backward");
  m.def("grid_grad", &monai::grid_grad, "GridGrad");
  m.def("grid_grad_backward", &monai::grid_grad_backward, "GridGrad backward");
}
