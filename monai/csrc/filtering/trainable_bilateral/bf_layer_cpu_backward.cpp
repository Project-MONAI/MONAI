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

=========================================================================
Adapted from https://github.com/faebstn96/trainable-bilateral-filter-source
which has the following license...
https://github.com/faebstn96/trainable-bilateral-filter-source/blob/main/LICENSE.md

Copyright 2022 Fabian Wagner, Pattern Recognition Lab, FAU Erlangen-Nuernberg, Erlangen, Germany
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

#include "trainable_bilateral.h"
#include "utils/tensor_description.h"
#include "utils/tensor_indexing.h"

template <typename scalar_t>
void BilateralFilterCpuBackward_3d(
    torch::Tensor gradientInputTensor,
    torch::Tensor gradientOutputTensor,
    torch::Tensor inputTensor,
    torch::Tensor outputTensor,
    torch::Tensor outputWeightsTensor,
    torch::Tensor dO_dx_ki,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma) {
  // Getting tensor description.
  TensorDescription desc = TensorDescription(gradientInputTensor);

  // Raw tensor data pointers.
  scalar_t* gradientInputTensorData = gradientInputTensor.data_ptr<scalar_t>();
  scalar_t* gradientOutputTensorData = gradientOutputTensor.data_ptr<scalar_t>();
  scalar_t* inputTensorData = inputTensor.data_ptr<scalar_t>();
  scalar_t* outputTensorData = outputTensor.data_ptr<scalar_t>();
  scalar_t* outputWeightsTensorData = outputWeightsTensor.data_ptr<scalar_t>();
  scalar_t* dO_dx_kiData = dO_dx_ki.data_ptr<scalar_t>();

  // Pre-calculate common values
  int windowSize_x = std::max(((int)ceil(5.0f * sigma_x) | 1), 5); // ORing last bit to ensure odd window size
  int windowSize_y = std::max(((int)ceil(5.0f * sigma_y) | 1), 5); // ORing last bit to ensure odd window size
  int windowSize_z = std::max(((int)ceil(5.0f * sigma_z) | 1), 5); // ORing last bit to ensure odd window size
  int halfWindowSize_x = floor(0.5f * windowSize_x);
  int halfWindowSize_y = floor(0.5f * windowSize_y);
  int halfWindowSize_z = floor(0.5f * windowSize_z);
  int halfWindowSize_arr[] = {halfWindowSize_x, halfWindowSize_y, halfWindowSize_z};
  scalar_t spatialExpConstant_x = -1.0f / (2 * sigma_x * sigma_x);
  scalar_t spatialExpConstant_y = -1.0f / (2 * sigma_y * sigma_y);
  scalar_t spatialExpConstant_z = -1.0f / (2 * sigma_z * sigma_z);
  scalar_t colorExpConstant = -1.0f / (2 * colorSigma * colorSigma);

  // Set kernel sizes with respect to the defined spatial sigmas.
  int* kernelSizes = new int[desc.dimensions];

  kernelSizes[0] = windowSize_x;
  kernelSizes[1] = windowSize_y;
  kernelSizes[2] = windowSize_z;

  // Pre-calculate gaussian kernel and distance map in 1D.
  scalar_t* gaussianKernel_x = new scalar_t[windowSize_x];
  scalar_t* gaussianKernel_y = new scalar_t[windowSize_y];
  scalar_t* gaussianKernel_z = new scalar_t[windowSize_z];
  scalar_t* xDistanceSquared = new scalar_t[windowSize_x];
  scalar_t* yDistanceSquared = new scalar_t[windowSize_y];
  scalar_t* zDistanceSquared = new scalar_t[windowSize_z];

  for (int i = 0; i < windowSize_x; i++) {
    int distance = i - halfWindowSize_x;
    gaussianKernel_x[i] = exp(distance * distance * spatialExpConstant_x);
    xDistanceSquared[i] = distance * distance;
  }
  for (int i = 0; i < windowSize_y; i++) {
    int distance = i - halfWindowSize_y;
    gaussianKernel_y[i] = exp(distance * distance * spatialExpConstant_y);
    yDistanceSquared[i] = distance * distance;
  }
  for (int i = 0; i < windowSize_z; i++) {
    int distance = i - halfWindowSize_z;
    gaussianKernel_z[i] = exp(distance * distance * spatialExpConstant_z);
    zDistanceSquared[i] = distance * distance;
  }

  // Looping over the batches
  for (int b = 0; b < desc.batchCount; b++) {
    int batchOffset = b * desc.batchStride;

    // Looping over all dimensions for the home element
    for (int z = 0; z < desc.sizes[2]; z++)
#pragma omp parallel for
      for (int y = 0; y < desc.sizes[1]; y++) {
        for (int x = 0; x < desc.sizes[0]; x++) {
          // Calculating indexing offset for the home element
          int homeOffset = batchOffset;

          int homeIndex[] = {x, y, z};
          homeOffset += x * desc.strides[0];
          homeOffset += y * desc.strides[1];
          homeOffset += z * desc.strides[2];

          // Zero kernel aggregates.
          scalar_t filter_kernel = 0;
          scalar_t valueSum = 0;

          // Looping over all dimensions for the neighbour element
          Indexer kernelIndex = Indexer(desc.dimensions, kernelSizes);
          do // while(kernelIndex++)
          {
            // Calculating buffer offset for the neighbour element
            // Index is clamped to the border in each dimension.
            int neighbourOffset = batchOffset;
            bool flagNotClamped = true;

            for (int i = 0; i < desc.dimensions; i++) {
              int neighbourIndex = homeIndex[i] + kernelIndex[i] - halfWindowSize_arr[i];
              int neighbourIndexClamped = std::min(desc.sizes[i] - 1, std::max(0, neighbourIndex));
              neighbourOffset += neighbourIndexClamped * desc.strides[i];
              if (neighbourIndex != neighbourIndexClamped) {
                flagNotClamped = false;
              }
            }

            // Euclidean color distance.
            scalar_t colorDistance = 0;
            scalar_t colorDistanceSquared = 0;

            for (int i = 0; i < desc.channelCount; i++) {
              scalar_t diff = inputTensorData[neighbourOffset + i * desc.channelStride] -
                  inputTensorData[homeOffset +
                                  i * desc.channelStride]; // Be careful: Here it is (X_k - X_i) and not (X_i - X_q)
              colorDistance += diff; // Do not take the absolute value here. Be careful with the signs.
              colorDistanceSquared += diff * diff;
            }

            // Calculating and combining the spatial
            // and color weights.
            scalar_t spatialWeight = 1;

            spatialWeight =
                gaussianKernel_x[kernelIndex[0]] * gaussianKernel_y[kernelIndex[1]] * gaussianKernel_z[kernelIndex[2]];

            scalar_t colorWeight = exp(colorDistanceSquared * colorExpConstant);
            scalar_t totalWeight = spatialWeight * colorWeight;

            // Aggregating values. Only do this if flagNotClamped: Pixels outside the image are disregarded.
            if (flagNotClamped) {
              for (int i = 0; i < desc.channelCount; i++) {
                // Distinguish cases for k!=i (calculation is done here)
                // and k==i (partial derivatives are precalculated).
                // If statement replaces center element of neighborhood/kernel.
                if (kernelIndex[0] != halfWindowSize_x || kernelIndex[1] != halfWindowSize_y ||
                    kernelIndex[2] != halfWindowSize_z) {
                  filter_kernel = -(1 / outputWeightsTensorData[neighbourOffset + i * desc.channelStride]) *
                          outputTensorData[neighbourOffset + i * desc.channelStride] * totalWeight * colorDistance /
                          (colorSigma * colorSigma) +
                      (1 / outputWeightsTensorData[neighbourOffset + i * desc.channelStride]) * totalWeight *
                          (1 +
                           inputTensorData[homeOffset + i * desc.channelStride] * colorDistance /
                               (colorSigma * colorSigma)); // inputTensorData[homeOffset] !!
                } else {
                  filter_kernel = dO_dx_kiData[homeOffset + i * desc.channelStride];
                }

                valueSum += gradientInputTensorData[neighbourOffset + i * desc.channelStride] * filter_kernel;
              }
            }
          } while (kernelIndex++);

          // Do the filtering and calculate the values for the backward pass.
          for (int i = 0; i < desc.channelCount; i++) {
            // Filtering:
            gradientOutputTensorData[homeOffset + i * desc.channelStride] = valueSum;
          }
        }
      }
  }

  delete[] kernelSizes;
  delete[] gaussianKernel_x;
  delete[] gaussianKernel_y;
  delete[] gaussianKernel_z;
  delete[] xDistanceSquared;
  delete[] yDistanceSquared;
  delete[] zDistanceSquared;
}

torch::Tensor BilateralFilterCpuBackward(
    torch::Tensor gradientInputTensor,
    torch::Tensor inputTensor,
    torch::Tensor outputTensor,
    torch::Tensor outputWeightsTensor,
    torch::Tensor dO_dx_ki,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma) {
  // Preparing output tensor.
  torch::Tensor gradientOutputTensor = torch::zeros_like(gradientInputTensor);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradientInputTensor.scalar_type(), "BilateralFilterCpuBackward_3d", ([&] {
                                        BilateralFilterCpuBackward_3d<scalar_t>(
                                            gradientInputTensor,
                                            gradientOutputTensor,
                                            inputTensor,
                                            outputTensor,
                                            outputWeightsTensor,
                                            dO_dx_ki,
                                            sigma_x,
                                            sigma_y,
                                            sigma_z,
                                            colorSigma);
                                      }));

  return gradientOutputTensor;
}
