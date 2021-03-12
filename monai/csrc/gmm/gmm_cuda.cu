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

#define BLOCK_SIZE 32
#define TILE(SIZE, STRIDE) ((((SIZE) - 1)/(STRIDE)) + 1)

#define CHANNELS 3
#define MAX_CHANNELS 16
#define MAX_MIXTURES 16


__constant__ int det_indices[] = { (9 << (4*4)) + (4 << (3*4)) + (6 << (2*4)) + (5 << (1*4)) + (4 << (0*4)),
    (5 << (4*4)) + (8 << (3*4)) + (6 << (2*4)) + (6 << (1*4)) + (7 << (0*4)),
    (5 << (4*4)) + (8 << (3*4)) + (7 << (2*4)) + (8 << (1*4)) + (9 << (0*4))
  };

__constant__ int inv_indices[] = { (4 << (5*4)) + (5 << (4*4)) + (4 << (3*4)) + (5 << (2*4)) + (6 << (1*4)) + (7 << (0*4)),
    (7 << (5*4)) + (6 << (4*4)) + (9 << (3*4)) + (8 << (2*4)) + (8 << (1*4)) + (9 << (0*4)),
    (5 << (5*4)) + (4 << (4*4)) + (6 << (3*4)) + (6 << (2*4)) + (5 << (1*4)) + (8 << (0*4)),
    (5 << (5*4)) + (8 << (4*4)) + (6 << (3*4)) + (7 << (2*4)) + (9 << (1*4)) + (8 << (0*4))
  };

__device__ __forceinline__ void self_outer_product_triangle(int length, float* vector, float* output, float diag_epsilon)
{
    for (int i = 0, x = 0; x < length; x++)
    {
        output[i] = vector[x] * vector[x] + diag_epsilon;
        i++;

        for (int y = x + 1; y < length; y++, i++)
        {
            output[i] = vector[x] * vector[y];
        }
    }
}

__device__ __forceinline__ float get_component(float* pixel, int i)
{
    switch (i)
    {
        case 0 : return 1.0f;
        case 1 : return pixel[0];
        case 2 : return pixel[1];
        case 3 : return pixel[2];
        case 4 : return pixel[0] * pixel[0];
        case 5 : return pixel[0] * pixel[1];
        case 6 : return pixel[0] * pixel[2];
        case 7 : return pixel[1] * pixel[1];
        case 8 : return pixel[1] * pixel[2];
        case 9 : return pixel[2] * pixel[2];
    };

    return 0.0f;
}

__device__ __forceinline__ float get_constant(float *gmm, int i)
{
    const float epsilon = 1.0e-3f;

    switch (i)
    {
        case 0 : return 0.0f;
        case 1 : return 0.0f;
        case 2 : return 0.0f;
        case 3 : return 0.0f;
        case 4 : return gmm[1] * gmm[1] + epsilon;
        case 5 : return gmm[1] * gmm[2];
        case 6 : return gmm[1] * gmm[3];
        case 7 : return gmm[2] * gmm[2] + epsilon;
        case 8 : return gmm[2] * gmm[3];
        case 9 : return gmm[3] * gmm[3] + epsilon;
    };

    return 0.0f;
}

template<int warp_count, int load_count>
__global__ void CovarianceReductionKernel(int gaussian_index, const float* g_image, const int* g_alpha, float* g_matrices, int element_count, int channel_count, int component_count, int mixture_count)
{
    __shared__ volatile float s_matrix_component[warp_count];

    const int block_size = warp_count << 5;

    int local_index = threadIdx.x;
    int block_index = blockIdx.x;
    int warp_index = local_index >> 5;
    int lane_index = local_index & 31;
    int global_index = local_index + block_index * block_size * load_count;
    int matrix_offset = (gaussian_index * gridDim.x + block_index) * component_count;

    float matrix[10];

    for (int i = 0; i < 10; i++)
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
                if (gaussian_index == (my_alpha & 15) + (my_alpha >> 4) * mixture_count)
                {
                    float pixel[3];
                    pixel[0] = g_image[global_index + 0 * element_count] * 255;
                    pixel[1] = g_image[global_index + 1 * element_count] * 255;
                    pixel[2] = g_image[global_index + 2 * element_count] * 255;
    
                    for (int i = 0; i < 10; i++)
                    {
                        matrix[i] += get_component(pixel, i);
                    }
                }
            }
        }
    }

    __syncthreads();

    for (int i = 0; i < 10; i++)
    {
        float matrix_component = matrix[i];

        matrix_component += __shfl_down_sync(0xffffffff, matrix_component, 16);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  8);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  4);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  2);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  1);

        if (lane_index == 0) { s_matrix_component[warp_index] = matrix_component; } __syncthreads();

        if (warp_index == 0) 
        { 
            if (warp_count >= 32) { s_matrix_component[lane_index] += s_matrix_component[lane_index + 16]; }
            if (warp_count >= 16) { s_matrix_component[lane_index] += s_matrix_component[lane_index +  8]; }
            if (warp_count >=  8) { s_matrix_component[lane_index] += s_matrix_component[lane_index +  4]; }
            if (warp_count >=  4) { s_matrix_component[lane_index] += s_matrix_component[lane_index +  2]; }
            if (warp_count >=  2) { s_matrix_component[lane_index] += s_matrix_component[lane_index +  1]; }

            if (lane_index == 0)
            {
                g_matrices[matrix_offset + i] = s_matrix_component[0];
            }
        }

        __syncthreads();
    }
}

template<int block_size, bool invert_sigma>
__global__ void CovarianceFinalizationKernel(const float* g_matrices, float* g_gmm, int matrix_count, int component_count)
{
    __shared__ volatile float s_matrix_component[block_size];
    __shared__ float s_gmm[15];

    int local_index = threadIdx.x;
    int gmm_index = blockIdx.x;
    int matrix_offset = gmm_index * matrix_count;
    
    int load_count = TILE(matrix_count, block_size);

    float matrix_component = 0.0f;
    float norm_factor = 1.0f;

    for (int i = 0; i < 10; i++)
    {
        matrix_component = 0.0f;

        for(int j = 0; j < load_count; j++)
        {
            int matrix_index = local_index + j * block_size;

            if(matrix_index < matrix_count)
            {
                matrix_component += g_matrices[(matrix_offset + matrix_index) * component_count + i];
            }
        }

        s_matrix_component[local_index] = matrix_component; __syncthreads();

        if (block_size >= 512) { if (local_index < 256) { s_matrix_component[local_index] += s_matrix_component[local_index + 256]; } __syncthreads(); }
        if (block_size >= 256) { if (local_index < 128) { s_matrix_component[local_index] += s_matrix_component[local_index + 128]; } __syncthreads(); }
        if (block_size >= 128) { if (local_index <  64) { s_matrix_component[local_index] += s_matrix_component[local_index +  64]; } __syncthreads(); }

        if (local_index <  32) 
        { 
            s_matrix_component[local_index] += s_matrix_component[local_index + 32];
            s_matrix_component[local_index] += s_matrix_component[local_index + 16];
            s_matrix_component[local_index] += s_matrix_component[local_index +  8];
            s_matrix_component[local_index] += s_matrix_component[local_index +  4];
            s_matrix_component[local_index] += s_matrix_component[local_index +  2];
            s_matrix_component[local_index] += s_matrix_component[local_index +  1];
        }

        __syncthreads();

        if (local_index == 0)
        {
            matrix_component = s_matrix_component[0];

            s_gmm[i] = norm_factor * matrix_component - get_constant(s_gmm, i);

            if (i == 0 && matrix_component > 0)
            {
                norm_factor = 1.0f / matrix_component;
            }
        }
    }

    if (local_index < 5)
    { 
        int idx0 = (det_indices[0] & (15 << (local_index * 4))) >> (local_index * 4);
        int idx1 = (det_indices[1] & (15 << (local_index * 4))) >> (local_index * 4);
        int idx2 = (det_indices[2] & (15 << (local_index * 4))) >> (local_index * 4);

        s_gmm[10 + local_index] = s_gmm[idx0] * s_gmm[idx1] * s_gmm[idx2];

        s_gmm[10] = s_gmm[10] + 2.0f * s_gmm[11] - s_gmm[12] - s_gmm[13] - s_gmm[14];
    }

    if (invert_sigma && local_index < 6)
    {
        int idx0 = (inv_indices[0] & (15 << (local_index * 4))) >> (local_index * 4);
        int idx1 = (inv_indices[1] & (15 << (local_index * 4))) >> (local_index * 4);
        int idx2 = (inv_indices[2] & (15 << (local_index * 4))) >> (local_index * 4);
        int idx3 = (inv_indices[3] & (15 << (local_index * 4))) >> (local_index * 4);

        if (s_gmm[10] > 0.0f)
        {
            s_gmm[4 + local_index] = (s_gmm[idx0] * s_gmm[idx1] - s_gmm[idx2] * s_gmm[idx3]) / s_gmm[10];
        }
        else
        {
            s_gmm[4 + local_index] = 0.0f;
        }
    }

    if (local_index < 11)
    {
        g_gmm[gmm_index * 11 + local_index] = s_gmm[local_index];
    }
}

// Single block, 32xmixture_count
__global__ void GMMcommonTerm(float *gmm, int mixture_count, int mixture_size, int component_count)
{
    int gmm_idx = (threadIdx.x * mixture_count) + threadIdx.y;

    float gmm_n = threadIdx.x < mixture_size ? gmm[gmm_idx * component_count] : 0.0f;

    float sum = gmm_n;

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum,  8);
    sum += __shfl_down_sync(0xffffffff, sum,  4);
    sum += __shfl_down_sync(0xffffffff, sum,  2);
    sum += __shfl_down_sync(0xffffffff, sum,  1);

    if (threadIdx.x < mixture_size)
    {
        float det = gmm[gmm_idx * component_count + 10];
        float commonTerm =  gmm_n / (sqrtf(det) * sum);

        gmm[gmm_idx * component_count + 10] = commonTerm;
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

__global__ void GMMDataTermKernel(const float *image, const float *gmm, float* output, int element_count, int mixture_count, int mixture_size, int component_count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= element_count) return;

    float temp_array[CHANNELS];
    temp_array[0] = image[index + 0 * element_count] * 255;
    temp_array[1] = image[index + 1 * element_count] * 255;
    temp_array[2] = image[index + 2 * element_count] * 255;

    float weights[MAX_MIXTURES];
    float weight_total = 0.0f;

    for(int i = 0; i < mixture_count; i++)
    {
        float mixture_weight = 0.0f;

        for(int j = 0; j < mixture_size; j++)
        {
            mixture_weight += GMMTerm(temp_array, &gmm[(mixture_count * j + i) * component_count]);
        }

        weights[i] = mixture_weight;
        weight_total += mixture_weight;
    }

    for(int i = 0; i < mixture_count; i++)
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

// 1 Block, 32xmixture_count
__global__ void GMMFindSplit(GMMSplit_t *gmmSplit, int gmmK, float *gmm, int channel_count, int component_count, int mixture_count)
{
    int gmm_idx = threadIdx.x * mixture_count + threadIdx.y;

    float eigenvalue = 0;
    float3 eigenvector;

    if (threadIdx.x < gmmK)
    {
        largest_eigenvalue_eigenvector(&gmm[gmm_idx * component_count + (channel_count + 1)], eigenvector, eigenvalue);
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
        split.threshold = scalar_prod(make_float3(gmm[gmm_idx * component_count + 1], gmm[gmm_idx * component_count + 2], gmm[gmm_idx * component_count + 3]), eigenvector);
        split.eigenvector = eigenvector;

        gmmSplit[threadIdx.y] = split;
    }
}

#define DO_SPLIT_DEGENERACY 4

__global__ void GMMDoSplit(const GMMSplit_t *gmmSplit, int k, float *gmm, int component_count, const float *image, int *alpha, int element_count)
{
    __shared__ GMMSplit_t s_gmmSplit[MAX_MIXTURES];

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
                    float temp_array[CHANNELS];
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

void GMMInitialize(const float *image, int *alpha, float *gmm, float *scratch_mem, int element_count, int mixture_count, int mixture_size, int channel_count, int component_count)
{
    int block_count = TILE(element_count, BLOCK * LOAD);
    
    float* block_gmm_scratch = scratch_mem;
    GMMSplit_t* gmm_split_scratch = (GMMSplit_t*) scratch_mem;

    int gmm_N = mixture_count * mixture_size;

    for (int k = mixture_count; k < gmm_N; k+=mixture_count)
    {
        for (int i = 0; i < k; ++i)
        {
            CovarianceReductionKernel<WARPS, LOAD><<<block_count, BLOCK>>>(i, image, alpha, block_gmm_scratch, element_count, channel_count, component_count, mixture_count);
        }

        CovarianceFinalizationKernel<THREADS, false><<<k, THREADS>>>(block_gmm_scratch, gmm, block_count, component_count);

        GMMFindSplit<<<1, dim3(BLOCK_SIZE, mixture_count)>>>(gmm_split_scratch, k / mixture_count, gmm, channel_count, component_count, mixture_count);
        GMMDoSplit<<<TILE(element_count, BLOCK_SIZE * DO_SPLIT_DEGENERACY), BLOCK_SIZE>>>(gmm_split_scratch, (k / mixture_count) << 4, gmm, component_count, image, alpha, element_count);
    }
}

void GMMUpdate(const float *image, int *alpha, float *gmm, float *scratch_mem, int element_count, int mixture_count, int mixture_size, int channel_count, int component_count)
{
    int block_count = TILE(element_count, BLOCK * LOAD);

    float* block_gmm_scratch = scratch_mem;

    int gmm_N = mixture_count * mixture_size;

    for (int i = 0; i < gmm_N; ++i)
    {
        CovarianceReductionKernel<WARPS, LOAD><<<block_count, BLOCK>>>(i, image, alpha, block_gmm_scratch, element_count, channel_count, component_count, mixture_count);
    }

    CovarianceFinalizationKernel<THREADS, true><<<gmm_N, THREADS>>>(block_gmm_scratch, gmm, block_count, component_count);

    GMMcommonTerm<<<1, dim3(BLOCK_SIZE, mixture_count)>>>(gmm, mixture_count, mixture_size, component_count);
}

void GMMDataTerm(const float *image, const float *gmm, float* output, int element_count, int mixture_count, int mixture_size, int component_count)
{
    dim3 block(BLOCK_SIZE, 1);
    dim3 grid(TILE(element_count, BLOCK_SIZE), 1);

    GMMDataTermKernel<<<grid, block>>>(image, gmm, output, element_count, mixture_count, mixture_size, component_count);
}

void GMM_Cuda(const float* input, const int* labels, float* output, int batch_count, int channel_count, int element_count, int mixture_count, int mixture_size)
{
    int component_count = 1 + (channel_count + 1) * (channel_count + 2) / 2;
    int gmm_size = component_count * mixture_count * mixture_size;

    float* scratch_mem = output;
    float* gmm; 
    int* alpha;

    cudaMalloc(&gmm, gmm_size * sizeof(float));
    cudaMalloc(&alpha, element_count * sizeof(int));

    cudaMemcpyAsync(alpha, labels, element_count * sizeof(int), cudaMemcpyDeviceToDevice);
    
    GMMInitialize(input, alpha, gmm, scratch_mem, element_count, mixture_count, mixture_size, channel_count, component_count);
    GMMUpdate(input, alpha, gmm, scratch_mem, element_count, mixture_count, mixture_size, channel_count, component_count);
    GMMDataTerm(input, gmm, output, element_count, mixture_count, mixture_size, component_count);

    cudaFree(alpha);
    cudaFree(gmm);
}
