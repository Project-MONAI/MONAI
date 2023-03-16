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
Adapted from https://github.com/faebstn96/trainable-joint-bilateral-filter-source
which has the following license...
https://github.com/faebstn96/trainable-joint-bilateral-filter-source/blob/main/LICENSE

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

#include "trainable_joint_bilateral.h"
#include "utils/tensor_description.h"
#include "utils/tensor_indexing.h"

template <typename scalar_t>
void JointBilateralFilterCpuForward_3d(
    torch::Tensor inputTensor,
    torch::Tensor guidanceTensor,
    torch::Tensor outputTensor,
    torch::Tensor outputWeightsTensor,
    torch::Tensor dO_dz_ki,
    torch::Tensor dO_dsig_r,
    torch::Tensor dO_dsig_x,
    torch::Tensor dO_dsig_y,
    torch::Tensor dO_dsig_z,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma) {
  // Getting tensor description.
  TensorDescription desc = TensorDescription(inputTensor);

  // Raw tensor data pointers.
  scalar_t* inputTensorData = inputTensor.data_ptr<scalar_t>();
  scalar_t* guidanceTensorData = guidanceTensor.data_ptr<scalar_t>();
  scalar_t* outputTensorData = outputTensor.data_ptr<scalar_t>();
  scalar_t* outputWeightsTensorData = outputWeightsTensor.data_ptr<scalar_t>();
  scalar_t* dO_dz_kiData = dO_dz_ki.data_ptr<scalar_t>();
  scalar_t* dO_dsig_rData = dO_dsig_r.data_ptr<scalar_t>();
  scalar_t* dO_dsig_xData = dO_dsig_x.data_ptr<scalar_t>();
  scalar_t* dO_dsig_yData = dO_dsig_y.data_ptr<scalar_t>();
  scalar_t* dO_dsig_zData = dO_dsig_z.data_ptr<scalar_t>();

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
          scalar_t valueSum = 0;
          scalar_t dw_dz_ki = 0;
          scalar_t dfilter_dz_ki = 0;
          scalar_t colorSum_w = 0;
          scalar_t colorSum_alpha = 0;
          scalar_t xSum_w = 0;
          scalar_t xSum_alpha = 0;
          scalar_t ySum_w = 0;
          scalar_t ySum_alpha = 0;
          scalar_t zSum_w = 0;
          scalar_t zSum_alpha = 0;

          scalar_t weightSum = 0.0f;

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
              scalar_t diff = guidanceTensorData[homeOffset + i * desc.channelStride] -
                  guidanceTensorData[neighbourOffset + i * desc.channelStride];
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
                valueSum += inputTensorData[neighbourOffset + i * desc.channelStride] * totalWeight;

                // Derivative of weights with respect to X_i while i=k.
                dw_dz_ki += (-1) * totalWeight * colorDistance / (colorSigma * colorSigma);
                // Derivative of convolved image with respect to X_i while i=k.
                dfilter_dz_ki += (-1) * totalWeight * inputTensorData[neighbourOffset + i * desc.channelStride] *
                    colorDistance /
                    (colorSigma *
                     colorSigma); // Be careful, the +1 is missing here -> Added before filling dfilter_dx_kiData

                colorSum_w += totalWeight * colorDistanceSquared / std::abs(colorSigma * colorSigma * colorSigma);
                colorSum_alpha += totalWeight * inputTensorData[neighbourOffset + i * desc.channelStride] *
                    colorDistanceSquared / std::abs(colorSigma * colorSigma * colorSigma);

                xSum_w += totalWeight * xDistanceSquared[kernelIndex[0]] / std::abs(sigma_x * sigma_x * sigma_x);
                xSum_alpha += totalWeight * inputTensorData[neighbourOffset + i * desc.channelStride] *
                    xDistanceSquared[kernelIndex[0]] / std::abs(sigma_x * sigma_x * sigma_x);

                ySum_w += totalWeight * yDistanceSquared[kernelIndex[1]] / std::abs(sigma_y * sigma_y * sigma_y);
                ySum_alpha += totalWeight * inputTensorData[neighbourOffset + i * desc.channelStride] *
                    yDistanceSquared[kernelIndex[1]] / std::abs(sigma_y * sigma_y * sigma_y);

                zSum_w += totalWeight * zDistanceSquared[kernelIndex[2]] / std::abs(sigma_z * sigma_z * sigma_z);
                zSum_alpha += totalWeight * inputTensorData[neighbourOffset + i * desc.channelStride] *
                    zDistanceSquared[kernelIndex[2]] / std::abs(sigma_z * sigma_z * sigma_z);
              }

              weightSum += totalWeight;
            }
          } while (kernelIndex++);

          // Do the filtering and calculate the values for the backward pass.
          for (int i = 0; i < desc.channelCount; i++) {
            // Filtering:
            outputTensorData[homeOffset + i * desc.channelStride] = valueSum / weightSum;

            // Pre-computations for the backward pass:
            outputWeightsTensorData[homeOffset + i * desc.channelStride] = weightSum;
            dO_dz_kiData[homeOffset + i * desc.channelStride] = -(1 / weightSum) * (valueSum / weightSum) * dw_dz_ki +
                (1 / weightSum) * (dfilter_dz_ki); // no +1 for dfilter_dz_ki for JBF added here!
            dO_dsig_rData[homeOffset + i * desc.channelStride] =
                -(1 / weightSum) * (valueSum / weightSum) * colorSum_w + (1 / weightSum) * colorSum_alpha;
            dO_dsig_xData[homeOffset + i * desc.channelStride] =
                -(1 / weightSum) * (valueSum / weightSum) * xSum_w + (1 / weightSum) * xSum_alpha;
            dO_dsig_yData[homeOffset + i * desc.channelStride] =
                -(1 / weightSum) * (valueSum / weightSum) * ySum_w + (1 / weightSum) * ySum_alpha;
            dO_dsig_zData[homeOffset + i * desc.channelStride] =
                -(1 / weightSum) * (valueSum / weightSum) * zSum_w + (1 / weightSum) * zSum_alpha;
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
JointBilateralFilterCpuForward(
    torch::Tensor inputTensor,
    torch::Tensor guidanceTensor,
    float sigma_x,
    float sigma_y,
    float sigma_z,
    float colorSigma) {
  // Preparing output tensor.
  torch::Tensor outputTensor = torch::zeros_like(inputTensor);
  torch::Tensor outputWeightsTensor = torch::zeros_like(inputTensor);
  torch::Tensor dO_dz_ki = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_r = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_x = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_y = torch::zeros_like(inputTensor);
  torch::Tensor dO_dsig_z = torch::zeros_like(inputTensor);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputTensor.scalar_type(), "JointBilateralFilterCpuForward_3d", ([&] {
                                        JointBilateralFilterCpuForward_3d<scalar_t>(
                                            inputTensor,
                                            guidanceTensor,
                                            outputTensor,
                                            outputWeightsTensor,
                                            dO_dz_ki,
                                            dO_dsig_r,
                                            dO_dsig_x,
                                            dO_dsig_y,
                                            dO_dsig_z,
                                            sigma_x,
                                            sigma_y,
                                            sigma_z,
                                            colorSigma);
                                      }));

  return {outputTensor, outputWeightsTensor, dO_dz_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z};
}
