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

/*
Adapted from https://github.com/abadams/permutohedral
which has the following license...

MIT License

Copyright (c) 2020 Andrew Adams

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#define BLOCK_SIZE 32

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>
#include <THC/THCAtomics.cuh>

#include "hash_table.cuh"
#include "permutohedral.h"
#include "utils/meta_macros.h"

template <typename scalar_t>
struct MatrixEntry {
  int index;
  scalar_t weight;
};

template <typename scalar_t, int pd>
__global__ static void createMatrix(
    const int elementCount,
    const scalar_t* positions,
    const scalar_t* values,
    const scalar_t* scaleFactor,
    MatrixEntry<scalar_t>* matrix) {
  const int threadId = threadIdx.x;
  const int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  const bool outOfBounds = idx >= elementCount;

  scalar_t myElevated[pd + 1];
  const scalar_t* myPosition = positions + idx * pd;

  int myGreedy[pd + 1];
  int myRank[pd + 1];

  scalar_t myBarycentric[pd + 2];
  __shared__ short keys[pd * BLOCK_SIZE];
  short* myKey = keys + threadId * pd;

  if (!outOfBounds) {
    myElevated[pd] = -pd * myPosition[pd - 1] * scaleFactor[pd - 1];

    for (int i = pd - 1; i > 0; i--) {
      myElevated[i] =
          myElevated[i + 1] - i * (myPosition[i - 1]) * scaleFactor[i - 1] + (i + 2) * myPosition[i] * scaleFactor[i];
    }

    myElevated[0] = myElevated[1] + 2 * myPosition[0] * scaleFactor[0];

    // find the closest zero-colored lattice point

    // greedily search for the closest zero-colored lattice point
    signed short sum = 0;

    for (int i = 0; i <= pd; i++) {
      scalar_t v = myElevated[i] * (1.0f / (pd + 1));
      scalar_t up = ceilf(v) * (pd + 1);
      scalar_t down = floorf(v) * (pd + 1);

      myGreedy[i] = (signed short)(up - myElevated[i] < myElevated[i] - down ? up : down);
      sum += myGreedy[i];
    }

    sum /= pd + 1;

    // sort differential to find the permutation between this simplex and the canonical one
    for (int i = 0; i <= pd; i++) {
      myRank[i] = 0;

      for (int j = 0; j <= pd; j++) {
        scalar_t iDiff = myElevated[i] - myGreedy[i];
        scalar_t jDiff = myElevated[j] - myGreedy[j];

        if (iDiff < jDiff || (iDiff == jDiff && i > j)) {
          myRank[i]++;
        }
      }
    }

    if (sum > 0) // sum too large, need to bring down the ones with the smallest differential
    {
      for (int i = 0; i <= pd; i++) {
        if (myRank[i] >= pd + 1 - sum) {
          myGreedy[i] -= (pd + 1);
          myRank[i] += sum - (pd + 1);
        } else {
          myRank[i] += sum;
        }
      }
    } else if (sum < 0) // sum too small, need to bring up the ones with largest differential
    {
      for (int i = 0; i <= pd; i++) {
        if (myRank[i] < -sum) {
          myGreedy[i] += (pd + 1);
          myRank[i] += sum + (pd + 1);
        } else {
          myRank[i] += sum;
        }
      }
    }

#ifdef LINEAR_D_MEMORY
    for (int i = 0; i <= pd; i++) {
      table_zeros[idx * (pd + 1) + i] = myGreedy[i];
      table_rank[idx * (pd + 1) + i] = myRank[i];
    }
#endif

    // turn delta into barycentric coords
    for (int i = 0; i <= pd + 1; i++) {
      myBarycentric[i] = 0;
    }

    for (int i = 0; i <= pd; i++) {
      scalar_t delta = (myElevated[i] - myGreedy[i]) * (1.0f / (pd + 1));
      myBarycentric[pd - myRank[i]] += delta;
      myBarycentric[pd + 1 - myRank[i]] -= delta;
    }

    myBarycentric[0] += 1.0f + myBarycentric[pd + 1];
  }

#ifdef USE_ADDITIVE_HASH
  unsigned int cumulative_hash = hash<pd>(myGreedy);
#endif

  for (int color = 0; color <= pd; color++) {
    // Compute the location of the lattice point explicitly (all but
    // the last coordinate - it's redundant because they sum to zero)
    if (!outOfBounds) {
      for (int i = 0; i < pd; i++) {
        myKey[i] = myGreedy[i] + color;

        if (myRank[i] > pd - color) {
          myKey[i] -= (pd + 1);
        }
      }
    }

#ifdef USE_ADDITIVE_HASH
    for (int i = 0; i < pd; i++) {
      if (myRank[i] == pd - color) {
        cumulative_hash += hOffset[i];
      }
    }
#endif

    if (!outOfBounds) {
      MatrixEntry<scalar_t> r;

#ifdef USE_ADDITIVE_HASH
      r.index = hashTableInsert<pd>(cumulative_hash, myKey, idx * (pd + 1) + color);
#else
      r.index = hashTableInsert<pd>(myKey, idx * (pd + 1) + color);
#endif

      r.weight = myBarycentric[color];
      matrix[idx * (pd + 1) + color] = r;
    }
  }
}

template <typename scalar_t, int kd>
__global__ static void cleanHashTable(const int elementCount, MatrixEntry<scalar_t>* matrix) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= elementCount)
    return;

  // find my hash table entry
  int* e = table_entries + idx;

  // Check if I created my own key in the previous phase
  if (*e >= 0) {
    // Rehash my key and reset the pointer in order to merge with
    // any other pixel that created a different entry under the
    // same key. If the computation was serial this would never
    // happen, but sometimes race conditions can make the same key
    // be inserted twice. hashTableRetrieve always returns the
    // earlier, so it's no problem as long as we rehash now.

#ifdef LINEAR_D_MEMORY
    // Get my key
    short myKey[kd];
    generateKey<kd>(*e, myKey);
    *e = hashTableRetrieve<kd>(myKey);
#else
    *e = hashTableRetrieve<kd>(table_keys + *e * kd);
#endif
  }
}

template <typename scalar_t, int pd, int vd>
__global__ static void splat(
    const int elementCount,
    scalar_t* values,
    MatrixEntry<scalar_t>* matrix,
    scalar_t* table_values) {
  const int color = threadIdx.y;
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const bool outOfBounds = idx >= elementCount;

  if (outOfBounds) {
    return;
  }

  scalar_t* myValue = values + idx * vd;

  MatrixEntry<scalar_t> r = matrix[idx * (pd + 1) + color];

  matrix[idx * (pd + 1) + color].index = r.index = table_entries[r.index];
  scalar_t* val = table_values + r.index * (vd + 1);

  for (int j = 0; j < vd; j++) {
    gpuAtomicAdd(val + j, myValue[j] * r.weight);
  }

  gpuAtomicAdd(val + vd, r.weight);
}

// splat splits by color, so extend the y coordinate to our blocks to represent that
// dim3 oldblocks((w-1)/8+1, (h-1)/8+1, 1);
// dim3 oldblockSize(8, 8, 1);
// oldblocks.y *= pd+1;
// splatCache<pd, vd><<<oldblocks, oldblockSize>>>(w, h, values, matrix);

// int blockCount = (elementCount + 1) / BLOCK_SIZE + 1;
// int blockSize = BLOCK_SIZE;

// splatCache<pd, vd><<<dim3(blockCount, 1), dim3(blockSize, pd+1)>>>(elementCount, values, matrix);

template <typename scalar_t, int pd, int vd>
__global__ static void splatCache(
    const int elementCount,
    scalar_t* values,
    MatrixEntry<scalar_t>* matrix,
    scalar_t* table_values) {
  // const int x = threadIdx.x + blockIdx.x * blockDim.x;
  // const int y = threadIdx.y + (blockIdx.y/(pd+1)) * blockDim.y;

  // const int threadId = threadIdx.y*blockDim.x + threadIdx.x;
  // const int color = blockIdx.y % (pd+1);
  // const int idx = y*w + x;

  const int threadId = threadIdx.x;
  const int color = threadIdx.y;
  const int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;

  const bool outOfBounds = idx >= elementCount;

  __shared__ int sharedOffsets[BLOCK_SIZE];
  __shared__ scalar_t sharedValues[BLOCK_SIZE * (vd + 1)];

  int myOffset = -1;
  scalar_t* myValue = sharedValues + threadId * (vd + 1);

  if (!outOfBounds) {
    scalar_t* value = values + idx * vd;

    MatrixEntry<scalar_t> r = matrix[idx * (pd + 1) + color];

    // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
    matrix[idx * (pd + 1) + color].index = r.index = table_entries[r.index];
    // record the offset into the keys/values array in shared space
    myOffset = sharedOffsets[threadId] = r.index * (vd + 1);

    for (int j = 0; j < vd; j++) {
      myValue[j] = value[j] * r.weight;
    }
    myValue[vd] = r.weight;

  } else {
    sharedOffsets[threadId] = -1;
  }

  __syncthreads();

  // am I the first thread in this block to care about this key?

  if (outOfBounds)
    return;

  for (int i = 0; i < BLOCK_SIZE; i++) {
    if (i < threadId) {
      if (myOffset == sharedOffsets[i]) {
        // somebody else with higher priority cares about this key
        return;
      }
    } else if (i > threadId) {
      if (myOffset == sharedOffsets[i]) {
        // someone else with lower priority cares about this key, accumulate it into mine
        for (int j = 0; j <= vd; j++) {
          sharedValues[threadId * (vd + 1) + j] += sharedValues[i * (vd + 1) + j];
        }
      }
    }
  }

  // only the threads with something to write to main memory are still going
  scalar_t* val = table_values + myOffset;
  for (int j = 0; j <= vd; j++) {
    gpuAtomicAdd(val + j, myValue[j]);
  }
}

template <typename scalar_t, int pd, int vd>
__global__ static void blur(
    int n,
    scalar_t* newValues,
    MatrixEntry<scalar_t>* matrix,
    int color,
    scalar_t* table_values) {
  const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

  if (idx >= n)
    return;

  // Check if I'm valid
  if (matrix[idx].index != idx)
    return;

  // find my key and the keys of my neighbours
  short myKey[pd + 1];
  short np[pd + 1];
  short nm[pd + 1];

#ifdef LINEAR_D_MEMORY
  generateKey<pd>(idx, myKey);
  for (int i = 0; i < pd; i++) {
    np[i] = myKey[i] + 1;
    nm[i] = myKey[i] - 1;
  }
#else
  for (int i = 0; i < pd; i++) {
    myKey[i] = table_keys[idx * pd + i];
    np[i] = myKey[i] + 1;
    nm[i] = myKey[i] - 1;
  }
#endif

  np[color] -= pd + 1;
  nm[color] += pd + 1;

#ifdef USE_ADDITIVE_HASH
  unsigned int hCurrent = hash<pd>(myKey);
  int offNp = hashTableRetrieveWithHash<pd>(hCurrent + hOffset[color], np);
  int offNm = hashTableRetrieveWithHash<pd>(hCurrent - hOffset[color], nm);
#else
  int offNp = hashTableRetrieve<pd>(np);
  int offNm = hashTableRetrieve<pd>(nm);
#endif

  scalar_t* valMe = table_values + (vd + 1) * idx;
  scalar_t* valNp = table_values + (vd + 1) * offNp;
  scalar_t* valNm = table_values + (vd + 1) * offNm;
  scalar_t* valOut = newValues + (vd + 1) * idx;

  if (offNp >= 0 && offNm >= 0) {
    for (int i = 0; i <= vd; i++) {
      valOut[i] = (valNp[i] + (valMe[i] * 2) + valNm[i]) / 4;
    }
  } else if (offNp >= 0) {
    for (int i = 0; i <= vd; i++) {
      valOut[i] = (valNp[i] + (valMe[i] * 2)) / 4;
    }
  } else if (offNm >= 0) {
    for (int i = 0; i <= vd; i++) {
      valOut[i] = (valNm[i] + (valMe[i] * 2)) / 4;
    }
  } else {
    for (int i = 0; i <= vd; i++) {
      valOut[i] = valMe[i] * 2;
    }
  }
}

template <typename scalar_t, int pd, int vd>
__global__ static void slice(
    const int elementCount,
    scalar_t* values,
    MatrixEntry<scalar_t>* matrix,
    scalar_t* table_values) {
  const int threadId = threadIdx.x;
  const int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  const bool outOfBounds = idx >= elementCount;

  if (outOfBounds)
    return;

  __shared__ scalar_t localValue[BLOCK_SIZE * vd];

  scalar_t* myValue = localValue + threadId * vd;
  scalar_t myWeight = 0;

  for (int i = 0; i < vd; i++) {
    myValue[i] = 0;
  }

  for (int i = 0; i <= pd; i++) {
    MatrixEntry<scalar_t> r = matrix[idx * (pd + 1) + i];
    scalar_t* val = table_values + r.index * (vd + 1);

    for (int j = 0; j < vd; j++) {
      myValue[j] += r.weight * val[j];
    }

    myWeight += r.weight * val[vd];
  }

  myWeight = 1.0f / myWeight;

  for (int j = 0; j < vd; j++) {
    values[idx * vd + j] = myValue[j] * myWeight;
  }
}

template <typename scalar_t, int vd, int pd>
void PermutohedralCuda(scalar_t* values, scalar_t* positions, int elementCount, bool accurate) {
  scalar_t blurVariance = accurate ? 0.5 : 0;

  scalar_t* scaleFactor;
  cudaMalloc(&scaleFactor, pd * sizeof(scalar_t));

  scalar_t scaleFactorHost[pd];
  for (int i = 0; i < pd; i++) {
    scaleFactorHost[i] = (pd + 1) * sqrtf((1.0 / 6 + blurVariance) / ((i + 1) * (i + 2)));
  }

  cudaMemcpy(scaleFactor, scaleFactorHost, pd * sizeof(scalar_t), cudaMemcpyHostToDevice);

  MatrixEntry<scalar_t>* matrix;
  cudaMalloc(&matrix, elementCount * (pd + 1) * sizeof(MatrixEntry<scalar_t>));

  scalar_t* table_values = createHashTable<scalar_t, pd, vd + 1>(elementCount * (pd + 1));

  // Populate constant memory for hash helpers
  unsigned long long int __host_two32 = ((unsigned long long int)1) << 32;
  unsigned int __host_div_c = 2 * (elementCount * (pd + 1));
  unsigned int __host_div_l = ceilf(logf((float)__host_div_c) / logf(2.0f));
  unsigned int __host_div_m = (__host_two32 << __host_div_l) / __host_div_c - __host_two32 + 1;
  cudaMemcpyToSymbol(__div_c, &__host_div_c, sizeof(unsigned int));
  cudaMemcpyToSymbol(__div_l, &__host_div_l, sizeof(unsigned int));
  cudaMemcpyToSymbol(__div_m, &__host_div_m, sizeof(unsigned int));

  // Populate constant memory with hash of offset vectors
  unsigned int hOffset_host[pd + 1];
  signed short offset[pd + 1];
  for (int i = 0; i < pd; offset[i] = 1, i++)
    ;
  for (int i = 0; i <= pd; i++) {
    offset[i] -= pd + 1;
    hOffset_host[i] = hash<pd>(offset);
    offset[i] += pd + 1;
  }
  cudaMemcpyToSymbol(hOffset, &hOffset_host, sizeof(unsigned int) * (pd + 1));

  int blockCount = (elementCount + 1) / BLOCK_SIZE + 1;
  int blockSize = BLOCK_SIZE;

  createMatrix<scalar_t, pd><<<blockCount, blockSize>>>(elementCount, positions, values, scaleFactor, matrix);

  // fix duplicate hash table entries
  int tableSize = elementCount * 2 * (pd + 1);
  int cleanBlockSize = 32;
  int cleanBlocks = (tableSize - 1) / cleanBlockSize + 1;

  cleanHashTable<scalar_t, pd><<<cleanBlocks, cleanBlockSize>>>(tableSize, matrix);

  splat<scalar_t, pd, vd><<<dim3(blockCount, 1), dim3(blockSize, pd + 1)>>>(elementCount, values, matrix, table_values);

  if (accurate) {
    scalar_t* newValues;
    cudaMalloc(&newValues, elementCount * (pd + 1) * (vd + 1) * sizeof(scalar_t));
    cudaMemset(newValues, 0, elementCount * (pd + 1) * (vd + 1) * sizeof(scalar_t));

    for (int color = 0; color <= pd; color++) {
      blur<scalar_t, pd, vd>
          <<<cleanBlocks, cleanBlockSize>>>(elementCount * (pd + 1), newValues, matrix, color, table_values);

      scalar_t* swap = newValues;
      newValues = table_values;
      table_values = swap;
    }

    cudaFree(newValues);
  }

  slice<scalar_t, pd, vd><<<blockCount, blockSize>>>(elementCount, values, matrix, table_values);

  destroyHashTable<scalar_t>();
  cudaFree(table_values);
  cudaFree(scaleFactor);
  cudaFree(matrix);
}

#define DECLARATION(dc, fc)                                                                                         \
  template void PermutohedralCuda<float, dc, fc>(float* values, float* positions, int elementCount, bool accurate); \
  template void PermutohedralCuda<double, dc, fc>(double* values, double* positions, int elementCount, bool accurate);
DO_FOR_AB(DECLARATION, 16, 19)
