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

// Struct to easily index input tensors.
struct Indexer {
 public:
  Indexer(int dimensions, int* sizes) {
    m_dimensions = dimensions;
    m_sizes = sizes;
    m_index = new int[dimensions]{0};
  }
  ~Indexer() {
    delete[] m_index;
  }

  bool operator++(int) {
    for (int i = 0; i < m_dimensions; i++) {
      m_index[i] += 1;

      if (m_index[i] < m_sizes[i]) {
        return true;
      } else {
        m_index[i] = 0;
      }
    }

    return false;
  }

  int& operator[](int dimensionIndex) {
    return m_index[dimensionIndex];
  }

 private:
  int m_dimensions;
  int* m_sizes;
  int* m_index;
};
