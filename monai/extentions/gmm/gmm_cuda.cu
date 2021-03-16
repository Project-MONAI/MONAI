/*
Copyright 2020 - 2021 MONAI Consortium
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

#include <cuda.h>
#include <cuda_runtime.h>

#include "gmm.h"
#include "gmm_cuda_linalg.cuh"

#define BLOCK_SIZE 32
#define TILE(SIZE, STRIDE) ((((SIZE) - 1)/(STRIDE)) + 1)

template<int warp_count, int load_count>
__global__ void CovarianceReductionKernel(int gaussian_index, const float* g_image, const int* g_alpha, float* g_matrices, int element_count)
{
    __shared__ float s_matrix_component[warp_count];

    const int block_size = warp_count << 5;

    int local_index = threadIdx.x;
    int block_index = blockIdx.x;
    int warp_index = local_index >> 5;
    int lane_index = local_index & 31;
    int global_index = local_index + block_index * block_size * load_count;
    int matrix_offset = (gaussian_index * gridDim.x + block_index) * GMM_COMPONENT_COUNT;

    float matrix[MATRIX_COMPONENT_COUNT];

    for (int i = 0; i < MATRIX_COMPONENT_COUNT; i++)
    {
        matrix[i] = 0;
    }

    for (int load = 0; load < load_count; load++)
    { 
        global_index += load * block_size;

        if (global_index < element_count)
        { 
            int my_alpha = g_alpha[global_index];
    
            if (my_alpha != -1)
            {
                if (gaussian_index == (my_alpha & 15) + (my_alpha >> 4) * MIXTURE_COUNT)
                {
                    float feature[CHANNEL_COUNT + 1];

                    feature[0] = 1;

                    for (int i = 0; i < CHANNEL_COUNT; i++)
                    {
                        feature[i + 1] = g_image[global_index + i * element_count] * 255;
                    }

                    for (int index = 0, i = 0; i < CHANNEL_COUNT + 1; i++)
                    {
                        for (int j = i; j < CHANNEL_COUNT + 1; j++, index++)
                        {
                            matrix[index] += feature[i] * feature[j];
                        }
                    }
                }
            }
        }
    }

    __syncthreads();

    for (int i = 0; i < MATRIX_COMPONENT_COUNT; i++)
    {
        float matrix_component = matrix[i];

        matrix_component += __shfl_down_sync(0xffffffff, matrix_component, 16);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  8);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  4);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  2);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  1);

        if (lane_index == 0)
        {
            s_matrix_component[warp_index] = matrix_component;
        }

        __syncthreads();

        if (warp_index == 0) 
        { 
            matrix_component = s_matrix_component[lane_index];

            if (warp_count >= 32) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component, 16); }
            if (warp_count >= 16) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  8); }
            if (warp_count >=  8) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  4); }
            if (warp_count >=  4) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  2); }
            if (warp_count >=  2) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  1); }

            if (lane_index == 0)
            {
                g_matrices[matrix_offset + i] = matrix_component;
            }
        }

        __syncthreads();
    }
}

template<int block_size, bool invert_matrix>
__global__ void CovarianceFinalizationKernel(const float* g_matrices, float* g_gmm, int matrix_count)
{
    __shared__ float s_matrix_component[block_size];
    __shared__ float s_gmm[15];

    int local_index = threadIdx.x;
    int gmm_index = blockIdx.x;
    int matrix_offset = gmm_index * matrix_count;
    
    int load_count = TILE(matrix_count, block_size);

    float norm_factor = 1.0f;

    for (int index = 0, i = 0; i < CHANNEL_COUNT + 1; i++)
    {
        for (int j = i; j < CHANNEL_COUNT + 1; j++, index++)
        {
            float matrix_component = 0.0f;

            for(int load = 0; load < load_count; load++)
            {
                int matrix_index = local_index + load * block_size;

                if(matrix_index < matrix_count)
                {
                    matrix_component += g_matrices[(matrix_offset + matrix_index) * GMM_COMPONENT_COUNT + index];
                }
            }

            s_matrix_component[local_index] = matrix_component; __syncthreads();

            if (block_size >= 512) { if (local_index < 256) { s_matrix_component[local_index] += s_matrix_component[local_index + 256]; } __syncthreads(); }
            if (block_size >= 256) { if (local_index < 128) { s_matrix_component[local_index] += s_matrix_component[local_index + 128]; } __syncthreads(); }
            if (block_size >= 128) { if (local_index <  64) { s_matrix_component[local_index] += s_matrix_component[local_index +  64]; } __syncthreads(); }

            if (local_index <  32)
            { 
                matrix_component = s_matrix_component[local_index];

                matrix_component += __shfl_down_sync(0xffffffff, matrix_component, 16);
                matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  8);
                matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  4);
                matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  2);
                matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  1);

                if (local_index == 0)
                {
                    float constant = i == 0 ? 0.0f : s_gmm[i] * s_gmm[j];

                    if (i != 0 && i == j)
                    {
                        constant += 1.0e-3f;
                    }

                    s_gmm[index] = norm_factor * matrix_component - constant;

                    if (index == 0 && matrix_component > 0)
                    {
                        norm_factor = 1.0f / matrix_component;
                    }
                }
            }

            __syncthreads();
        }
    }

    float* matrix = s_gmm + (CHANNEL_COUNT + 1);
    float* det_ptr = s_gmm + MATRIX_COMPONENT_COUNT;

    CalculateDeterminant(matrix, det_ptr, local_index);

    if (invert_matrix)
    {
        InvertMatrix(matrix, *det_ptr, local_index);
    }

    if (local_index < MATRIX_COMPONENT_COUNT + 1)
    {
        g_gmm[gmm_index * (MATRIX_COMPONENT_COUNT + 1) + local_index] = s_gmm[local_index];
    }
}

// Single block, 32xMIXTURE_COUNT
__global__ void GMMcommonTerm(float *gmm)
{
    int gmm_idx = (threadIdx.x * MIXTURE_COUNT) + threadIdx.y;

    float gmm_n = threadIdx.x < MIXTURE_SIZE ? gmm[gmm_idx * GMM_COMPONENT_COUNT] : 0.0f;

    float sum = gmm_n;

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum,  8);
    sum += __shfl_down_sync(0xffffffff, sum,  4);
    sum += __shfl_down_sync(0xffffffff, sum,  2);
    sum += __shfl_down_sync(0xffffffff, sum,  1);

    if (threadIdx.x < MIXTURE_SIZE)
    {
        float det = gmm[gmm_idx * GMM_COMPONENT_COUNT + 10];
        float commonTerm =  gmm_n / (sqrtf(det) * sum);

        gmm[gmm_idx * GMM_COMPONENT_COUNT + 10] = commonTerm;
    }
}

__device__ float GMMTerm(float* pixel, const float *gmm)
{
    float3 v = make_float3(pixel[0] - gmm[1], pixel[1] - gmm[2], pixel[2] - gmm[3]);

    float xxa = v.x * v.x * gmm[4];
    float yyd = v.y * v.y * gmm[7];
    float zzf = v.z * v.z * gmm[9];

    float yxb = v.x * v.y * gmm[5];
    float zxc = v.z * v.x * gmm[6];
    float zye = v.z * v.y * gmm[8];

    return gmm[10] * expf(-0.5f * (xxa + yyd + zzf + 2.0f * (yxb + zxc + zye)));
}

__global__ void GMMDataTermKernel(const float *image, const float *gmm, float* output, int element_count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= element_count) return;

    float temp_array[CHANNEL_COUNT];
    temp_array[0] = image[index + 0 * element_count] * 255;
    temp_array[1] = image[index + 1 * element_count] * 255;
    temp_array[2] = image[index + 2 * element_count] * 255;

    float weights[MIXTURE_COUNT];
    float weight_total = 0.0f;

    for(int i = 0; i < MIXTURE_COUNT; i++)
    {
        float mixture_weight = 0.0f;

        for(int j = 0; j < MIXTURE_SIZE; j++)
        {
            mixture_weight += GMMTerm(temp_array, &gmm[(MIXTURE_COUNT * j + i) * GMM_COMPONENT_COUNT]);
        }

        weights[i] = mixture_weight;
        weight_total += mixture_weight;
    }

    for(int i = 0; i < MIXTURE_COUNT; i++)
    {
        output[index + i * element_count] = weights[i] / weight_total;
    }
}

__device__
float3 normalize(float3 v)
{
    float norm = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

    return make_float3(v.x * norm, v.y * norm, v.z * norm);
}

__device__
float3 mul_right(const float *M, float3 v)
{
    return make_float3(
               M[0] * v.x + M[1] * v.y + M[2] * v.z,
               M[1] * v.x + M[3] * v.y + M[4] * v.z,
               M[2] * v.x + M[4] * v.y + M[5] * v.z);
}

__device__
float largest_eigenvalue(const float *M)
{
    float norm = M[0] > M[3] ? M[0] : M[3];
    norm = M[0] > M[5] ? M[0] : M[5];
    norm = 1.0f / norm;

    float a00 = norm * M[0];
    float a01 = norm * M[1];
    float a02 = norm * M[2];
    float a11 = norm * M[3];
    float a12 = norm * M[4];
    float a22 = norm * M[5];

    float c0 = a00*a11*a22 + 2.0f*a01*a02*a12 - a00*a12*a12 - a11*a02*a02 - a22*a01*a01;
    float c1 = a00*a11 - a01*a01 + a00*a22 - a02*a02 + a11*a22 - a12*a12;
    float c2 = a00 + a11 + a22;

    const float inv3 = 1.0f / 3.0f;
    const float root3 = sqrtf(3.0f);

    float c2Div3 = c2*inv3;
    float aDiv3 = (c1 - c2*c2Div3)*inv3;

    if (aDiv3 > 0.0f)
    {
        aDiv3 = 0.0f;
    }

    float mbDiv2 = 0.5f*(c0 + c2Div3*(2.0f*c2Div3*c2Div3 - c1));
    float q = mbDiv2*mbDiv2 + aDiv3*aDiv3*aDiv3;

    if (q > 0.0f)
    {
        q = 0.0f;
    }

    float magnitude = sqrtf(-aDiv3);
    float angle = atan2(sqrtf(-q),mbDiv2)*inv3;
    float cs = cos(angle);
    float sn = sin(angle);

    float largest_eigenvalue = c2Div3 + 2.0f*magnitude*cs;

    float eigenvalue = c2Div3 - magnitude*(cs + root3*sn);

    if (eigenvalue > largest_eigenvalue)
    {
        largest_eigenvalue = eigenvalue;
    }

    eigenvalue = c2Div3 - magnitude*(cs - root3*sn);

    if (eigenvalue > largest_eigenvalue)
    {
        largest_eigenvalue = eigenvalue;
    }

    return largest_eigenvalue / norm;
}

__device__
float3 cross_prod(float3 a, float3 b)
{
    return make_float3((a.y*b.z)-(a.z*b.y), (a.z*b.x)-(a.x*b.z), (a.x*b.y)-(a.y*b.x));
}

__device__
float3 compute_eigenvector(const float *M, float eigenvalue)
{
    float3 r0 = make_float3(M[0] - eigenvalue, M[1], M[2]);
    float3 r1 = make_float3(M[2] , M[3]- eigenvalue, M[4]);

    float3 eigenvector = cross_prod(r0,r1);
    return normalize(eigenvector);
}

__device__
void largest_eigenvalue_eigenvector(const float *M, float3 &evec, float &eval)
{
    eval = largest_eigenvalue(M);
    evec = compute_eigenvector(M, eval);
}

__device__
float scalar_prod(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct GMMSplit_t
{
    int idx;
    float threshold;
    float3 eigenvector;
};

// 1 Block, 32xMIXTURE_COUNT
__global__ void GMMFindSplit(GMMSplit_t *gmmSplit, int gmmK, float *gmm)
{
    int gmm_idx = threadIdx.x * MIXTURE_COUNT + threadIdx.y;

    float eigenvalue = 0;
    float3 eigenvector;

    if (threadIdx.x < gmmK)
    {
        largest_eigenvalue_eigenvector(&gmm[gmm_idx * GMM_COMPONENT_COUNT + (CHANNEL_COUNT + 1)], eigenvector, eigenvalue);
    }

    float max_value = eigenvalue;

    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 16));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value,  8));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value,  4));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value,  2));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value,  1));

    if (max_value == eigenvalue)
    {
        GMMSplit_t split;

        split.idx = threadIdx.x;
        split.threshold = scalar_prod(make_float3(gmm[gmm_idx * GMM_COMPONENT_COUNT + 1], gmm[gmm_idx * GMM_COMPONENT_COUNT + 2], gmm[gmm_idx * GMM_COMPONENT_COUNT + 3]), eigenvector);
        split.eigenvector = eigenvector;

        gmmSplit[threadIdx.y] = split;
    }
}

#define DO_SPLIT_DEGENERACY 4

__global__ void GMMDoSplit(const GMMSplit_t *gmmSplit, int k, float *gmm, const float *image, int *alpha, int element_count)
{
    __shared__ GMMSplit_t s_gmmSplit[MIXTURE_COUNT];

    int *s_linear = (int *) s_gmmSplit;
    int *g_linear = (int *) gmmSplit;

    if (threadIdx.x < 10)
    {
        s_linear[threadIdx.x] = g_linear[threadIdx.x];
    }

    __syncthreads();

    int index = threadIdx.x + blockIdx.x * BLOCK_SIZE * DO_SPLIT_DEGENERACY;

    for (int i = 0; i < DO_SPLIT_DEGENERACY; i++)
    {
        index += BLOCK_SIZE;

        if (index < element_count)
        {
            int my_alpha = alpha[index];

            if(my_alpha != -1)
            {
                int select = my_alpha & 15;
                int gmm_idx = my_alpha >> 4;
    
                if (gmm_idx == s_gmmSplit[select].idx)
                {
                    // in the split cluster now
                    float temp_array[CHANNEL_COUNT];
                    temp_array[0] = image[index + 0 * element_count] * 255;
                    temp_array[1] = image[index + 1 * element_count] * 255;
                    temp_array[2] = image[index + 2 * element_count] * 255;
    
                    float value = scalar_prod(s_gmmSplit[select].eigenvector, make_float3(temp_array[0], temp_array[1], temp_array[2]));
    
                    if (value > s_gmmSplit[select].threshold)
                    {
                        // assign pixel to new cluster
                        alpha[index] =  k + select;
                    }
                }
            }
        }
    }
}

#define THREADS 512
#define WARPS 16
#define BLOCK (WARPS << 5)
#define LOAD 4

void GMMInitialize(const float *image, int *alpha, float *gmm, float *scratch_mem, int element_count)
{
    int block_count = TILE(element_count, BLOCK * LOAD);
    
    float* block_gmm_scratch = scratch_mem;
    GMMSplit_t* gmm_split_scratch = (GMMSplit_t*) scratch_mem;

    int gmm_N = MIXTURE_COUNT * MIXTURE_SIZE;

    for (int k = MIXTURE_COUNT; k < gmm_N; k+=MIXTURE_COUNT)
    {
        for (int i = 0; i < k; ++i)
        {
            CovarianceReductionKernel<WARPS, LOAD><<<block_count, BLOCK>>>(i, image, alpha, block_gmm_scratch, element_count);
        }

        CovarianceFinalizationKernel<THREADS, false><<<k, THREADS>>>(block_gmm_scratch, gmm, block_count);

        GMMFindSplit<<<1, dim3(BLOCK_SIZE, MIXTURE_COUNT)>>>(gmm_split_scratch, k / MIXTURE_COUNT, gmm);
        GMMDoSplit<<<TILE(element_count, BLOCK_SIZE * DO_SPLIT_DEGENERACY), BLOCK_SIZE>>>(gmm_split_scratch, (k / MIXTURE_COUNT) << 4, gmm, image, alpha, element_count);
    }
}

void GMMUpdate(const float *image, int *alpha, float *gmm, float *scratch_mem, int element_count)
{
    int block_count = TILE(element_count, BLOCK * LOAD);

    float* block_gmm_scratch = scratch_mem;

    int gmm_N = MIXTURE_COUNT * MIXTURE_SIZE;

    for (int i = 0; i < gmm_N; ++i)
    {
        CovarianceReductionKernel<WARPS, LOAD><<<block_count, BLOCK>>>(i, image, alpha, block_gmm_scratch, element_count);
    }

    CovarianceFinalizationKernel<THREADS, true><<<gmm_N, THREADS>>>(block_gmm_scratch, gmm, block_count);

    GMMcommonTerm<<<1, dim3(BLOCK_SIZE, MIXTURE_COUNT)>>>(gmm);
}

void GMMDataTerm(const float *image, const float *gmm, float* output, int element_count)
{
    dim3 block(BLOCK_SIZE, 1);
    dim3 grid(TILE(element_count, BLOCK_SIZE), 1);

    GMMDataTermKernel<<<grid, block>>>(image, gmm, output, element_count);
}

void GMM_Cuda(const float* input, const int* labels, float* output, int batch_count, int element_count)
{
    float* scratch_mem = output;
    float* gmm; 
    int* alpha;

    cudaMalloc(&gmm, GMM_COUNT * GMM_COMPONENT_COUNT * sizeof(float));
    cudaMalloc(&alpha, element_count * sizeof(int));

    cudaMemcpyAsync(alpha, labels, element_count * sizeof(int), cudaMemcpyDeviceToDevice);
    
    GMMInitialize(input, alpha, gmm, scratch_mem, element_count);
    GMMUpdate(input, alpha, gmm, scratch_mem, element_count);
    GMMDataTerm(input, gmm, output, element_count);

    cudaFree(alpha);
    cudaFree(gmm);
}
