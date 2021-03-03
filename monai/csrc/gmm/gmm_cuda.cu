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


// Tile Size: 32x32, Block Size 32xwarp_N
template<int warp_N, bool create_gmm_flags>
__global__ void GMMReductionKernel(int gmm_idx, const float* image, const int* alpha, float* gmm, unsigned int* tile_gmms, int element_count, int component_count, int mixture_count)
{
    __shared__ float s_lists[32 * 32 * CHANNELS];
    __shared__ volatile float s_gmm[32 * warp_N];
    __shared__ float s_final[warp_N];

    __shared__ int gmm_flags[32];

    int warp_idx = threadIdx.x >> 5;
    int thread_idx = threadIdx.x;
    int lane_idx = threadIdx.x & 31;

    int block_idx = blockIdx.x;

    float *block_gmm = &gmm[(gridDim.x * gmm_idx + block_idx) * component_count];
    volatile float *warp_gmm = &s_gmm[warp_idx * 32];

    if (create_gmm_flags)
    {
        if (warp_idx == 0)
        {
            gmm_flags[lane_idx] = 0;
        }

        __syncthreads();
    }
    else
    {
        unsigned int gmm_mask = tile_gmms[block_idx];

        if ((gmm_mask & (1u << gmm_idx)) == 0)
        {

            if (lane_idx < 10 && warp_idx == 0)
            {
                block_gmm[lane_idx] = 0.0f;
            }

            return;
        }
    }

    int list_idx = 0;

    int index = block_idx * 32 * 32 + warp_idx * 32 + lane_idx;

    // Build lists of pixels that belong to this GMM

    for (int k=0; k < (32/warp_N); ++k)
    {
        if (index < element_count)
        {
            int my_alpha = alpha[index];

            if (my_alpha != -1)
            {
                int my_gmm_idx = (my_alpha & 15) + (my_alpha >> 4) * mixture_count;

                if (create_gmm_flags)
                {
                    gmm_flags[my_gmm_idx] = 1;
                }
    
                if (my_gmm_idx == gmm_idx)
                {
                    s_lists[(thread_idx + list_idx * (32*warp_N)) * CHANNELS + 0] = image[index + 0 * element_count] * 255;
                    s_lists[(thread_idx + list_idx * (32*warp_N)) * CHANNELS + 1] = image[index + 1 * element_count] * 255;
                    s_lists[(thread_idx + list_idx * (32*warp_N)) * CHANNELS + 2] = image[index + 2 * element_count] * 255;
                    ++list_idx;
                }
            }
        }

        index += warp_N * 32;
    }

    __syncthreads();

    if (warp_idx == 0 && create_gmm_flags)
    {
        tile_gmms[block_idx] = __ballot_sync(0xFFFFFFFF, gmm_flags[lane_idx] > 0);
    }

    // Reduce for each global GMM element

    for (int i=0; i<10; ++i)
    {
        float thread_gmm;

        if (i == 0)
        {
            // thread_gmm = list_idx for first component
            thread_gmm = list_idx;
        }
        else
        {
            float temp_array[CHANNELS];
            temp_array[0] = s_lists[thread_idx * CHANNELS + 0];
            temp_array[1] = s_lists[thread_idx * CHANNELS + 1];
            temp_array[2] = s_lists[thread_idx * CHANNELS + 2];

            thread_gmm = list_idx > 0 ? get_component(temp_array, i) : 0.0f;

            for (int k=1; k<(32/warp_N) && k < list_idx; ++k)
            {
                temp_array[0] = s_lists[(thread_idx + k * (32*warp_N)) * CHANNELS + 0];
                temp_array[1] = s_lists[(thread_idx + k * (32*warp_N)) * CHANNELS + 1];
                temp_array[2] = s_lists[(thread_idx + k * (32*warp_N)) * CHANNELS + 2];

                thread_gmm += get_component(temp_array, i);
            }
        }

        warp_gmm[lane_idx] = thread_gmm;

        // Warp Reductions
        thread_gmm += warp_gmm[(lane_idx + 16) & 31];
        warp_gmm[lane_idx] = thread_gmm;

        thread_gmm += warp_gmm[(lane_idx + 8) & 31];
        warp_gmm[lane_idx] = thread_gmm;

        thread_gmm += warp_gmm[(lane_idx + 4) & 31];
        warp_gmm[lane_idx] = thread_gmm;

        thread_gmm += warp_gmm[(lane_idx + 2) & 31];
        warp_gmm[lane_idx] = thread_gmm;

        thread_gmm += warp_gmm[(lane_idx + 1) & 31];
        s_final[warp_idx] = thread_gmm;

        __syncthreads();

        // Final Reduction
        if (warp_idx == 0 && lane_idx == 0)
        {
            for (int j=1; j<warp_N; ++j)
            {
                thread_gmm += s_final[j];
            }

            block_gmm[i] = thread_gmm;
        }
    }
}

__constant__ int det_indices[] = { (9 << (4*4)) + (4 << (3*4)) + (6 << (2*4)) + (5 << (1*4)) + (4 << (0*4)),
                                   (5 << (4*4)) + (8 << (3*4)) + (6 << (2*4)) + (6 << (1*4)) + (7 << (0*4)),
                                   (5 << (4*4)) + (8 << (3*4)) + (7 << (2*4)) + (8 << (1*4)) + (9 << (0*4))
                                 };

__constant__ int inv_indices[] = { (4 << (5*4)) + (5 << (4*4)) + (4 << (3*4)) + (5 << (2*4)) + (6 << (1*4)) + (7 << (0*4)),
                                   (7 << (5*4)) + (6 << (4*4)) + (9 << (3*4)) + (8 << (2*4)) + (8 << (1*4)) + (9 << (0*4)),
                                   (5 << (5*4)) + (4 << (4*4)) + (6 << (3*4)) + (6 << (2*4)) + (5 << (1*4)) + (8 << (0*4)),
                                   (5 << (5*4)) + (8 << (4*4)) + (6 << (3*4)) + (7 << (2*4)) + (9 << (1*4)) + (8 << (0*4))
                                 };


// One block per GMM, 32*warp_N threads (1-dim)
template <int warp_N, bool invertSigma>
__global__ void GMMFinalizeKernel(const float *gmm_scratch, float *gmm, int gmm_stride, int component_count)
{
    __shared__ volatile float s_gmm[warp_N*32];
    __shared__ float s_final[warp_N];
    __shared__ float final_gmm[15];

    const int thread_N = warp_N * 32;

    const float *gmm_partial = &gmm_scratch[blockIdx.x * gmm_stride * component_count];

    volatile float *warp_gmm = &s_gmm[threadIdx.x & 0x0ffe0];

    int thread_idx = threadIdx.x;
    int lane_idx = threadIdx.x & 31;
    int warp_idx = threadIdx.x >> 5;

    float norm_factor = 1.0f;

    for (int i=0; i<10; ++i)
    {
        float thread_gmm = 0.0f;

        for (int j=thread_idx; j < gmm_stride; j+= thread_N)
        {
            thread_gmm += gmm_partial[j * component_count + i];
        }

        warp_gmm[lane_idx] = thread_gmm;

        // Warp Reduction
        thread_gmm += warp_gmm[(lane_idx + 16) & 31];
        warp_gmm[lane_idx] = thread_gmm;

        thread_gmm += warp_gmm[(lane_idx + 8) & 31];
        warp_gmm[lane_idx] = thread_gmm;

        thread_gmm += warp_gmm[(lane_idx + 4) & 31];
        warp_gmm[lane_idx] = thread_gmm;

        thread_gmm += warp_gmm[(lane_idx + 2) & 31];
        warp_gmm[lane_idx] = thread_gmm;

        thread_gmm += warp_gmm[(lane_idx + 1) & 31];

        s_final[warp_idx] = thread_gmm;

        __syncthreads();

        // Final Reduction
        if (warp_idx == 0 && lane_idx == 0)
        {
            for (int j=1; j<warp_N; ++j)
            {
                thread_gmm += s_final[j];
            }

            final_gmm[i] = norm_factor * thread_gmm - get_constant(final_gmm, i);

            if (i == 0)
            {
                if (thread_gmm > 0)
                {
                    norm_factor = 1.0f / thread_gmm;
                }
            }
        }
    }

    if (threadIdx.y == 0)
    {
        // Compute det(Sigma) using final_gmm [10-14] as scratch mem

        if (threadIdx.x < 5)
        {

            int idx0 = (det_indices[0] & (15 << (threadIdx.x * 4))) >> (threadIdx.x * 4);
            int idx1 = (det_indices[1] & (15 << (threadIdx.x * 4))) >> (threadIdx.x * 4);
            int idx2 = (det_indices[2] & (15 << (threadIdx.x * 4))) >> (threadIdx.x * 4);

            final_gmm[10 + threadIdx.x] = final_gmm[idx0] * final_gmm[idx1] * final_gmm[idx2];

            float det = final_gmm[10] + 2.0f * final_gmm[11] - final_gmm[12] - final_gmm[13] - final_gmm[14];
            final_gmm[10] = det;
        }

        // Compute inv(Sigma)
        if (invertSigma && threadIdx.x < 6)
        {
            int idx0 = (inv_indices[0] & (15 << (threadIdx.x * 4))) >> (threadIdx.x * 4);
            int idx1 = (inv_indices[1] & (15 << (threadIdx.x * 4))) >> (threadIdx.x * 4);
            int idx2 = (inv_indices[2] & (15 << (threadIdx.x * 4))) >> (threadIdx.x * 4);
            int idx3 = (inv_indices[3] & (15 << (threadIdx.x * 4))) >> (threadIdx.x * 4);

            float temp = final_gmm[idx0] * final_gmm[idx1] - final_gmm[idx2] * final_gmm[idx3];

            if (final_gmm[10] > 0.0f)
            {
                final_gmm[4+threadIdx.x] = temp / final_gmm[10];
            }
            else
            {
                final_gmm[4+threadIdx.x] = 0.0f;
            }
        }

        if (threadIdx.x < 11)
        {
            gmm[blockIdx.x * component_count + threadIdx.x] = final_gmm[threadIdx.x];
        }
    }
}


// Single block, 32x2
__global__ void GMMcommonTerm(float *gmm, int mixture_count, int mixture_size, int component_count)
{
    __shared__ volatile float s_n[MAX_MIXTURES][32];

    int gmm_idx = (threadIdx.x * mixture_count) | threadIdx.y;

    float gmm_n = threadIdx.x < mixture_size ? gmm[gmm_idx * component_count] : 0.0f;
    float sum = gmm_n;
    s_n[threadIdx.y][threadIdx.x] = sum;

    // Warp Reduction
    sum += s_n[threadIdx.y][(threadIdx.x + 16) & 31];
    s_n[threadIdx.y][threadIdx.x] = sum;

    sum += s_n[threadIdx.y][(threadIdx.x + 8) & 31];
    s_n[threadIdx.y][threadIdx.x] = sum;

    sum += s_n[threadIdx.y][(threadIdx.x + 4) & 31];
    s_n[threadIdx.y][threadIdx.x] = sum;

    sum += s_n[threadIdx.y][(threadIdx.x + 2) & 31];
    s_n[threadIdx.y][threadIdx.x] = sum;

    sum += s_n[threadIdx.y][(threadIdx.x + 1) & 31];

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
__global__ void GMMFindSplit(GMMSplit_t *gmmSplit, int gmmK, float *gmm, int component_count, int mixture_count)
{
    __shared__ float s_eigenvalues[MAX_MIXTURES][32];

    // int gmm_idx = (threadIdx.x << 1) + threadIdx.y;
    int gmm_idx = threadIdx.x * mixture_count + threadIdx.y;

    float eigenvalue = 0;
    float3 eigenvector;

    if (threadIdx.x < gmmK)
    {
        largest_eigenvalue_eigenvector(&gmm[gmm_idx * component_count + 4], eigenvector, eigenvalue);
    }

    // Warp Reduction
    float maxvalue = eigenvalue;
    s_eigenvalues[threadIdx.y][threadIdx.x] = maxvalue;

    maxvalue = max(maxvalue, s_eigenvalues[threadIdx.y][(threadIdx.x+16) & 31]);
    s_eigenvalues[threadIdx.y][threadIdx.x] = maxvalue;

    maxvalue = max(maxvalue, s_eigenvalues[threadIdx.y][(threadIdx.x+8) & 31]);
    s_eigenvalues[threadIdx.y][threadIdx.x] = maxvalue;

    maxvalue = max(maxvalue, s_eigenvalues[threadIdx.y][(threadIdx.x+4) & 31]);
    s_eigenvalues[threadIdx.y][threadIdx.x] = maxvalue;

    maxvalue = max(maxvalue, s_eigenvalues[threadIdx.y][(threadIdx.x+2) & 31]);
    s_eigenvalues[threadIdx.y][threadIdx.x] = maxvalue;

    maxvalue = max(maxvalue, s_eigenvalues[threadIdx.y][(threadIdx.x+1) & 31]);

    if (maxvalue == eigenvalue)
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

void GMMInitialize(const float *image, int *alpha, float *gmm, float *scratch_mem, int element_count, int mixture_count, int mixture_size, int component_count)
{
    dim3 grid(TILE(element_count, BLOCK_SIZE * BLOCK_SIZE));
    dim3 block(BLOCK_SIZE * 4);
    
    float* block_gmm_scratch = &scratch_mem[grid.x];
    unsigned int* block_active_scratch = (unsigned int*)scratch_mem;
    GMMSplit_t* gmm_split_scratch = (GMMSplit_t*) scratch_mem;

    int gmm_N = mixture_count * mixture_size;

    for (int k = mixture_count; k < gmm_N; k+=mixture_count)
    {
        GMMReductionKernel<4, true><<<grid, block>>>(0, image, alpha, block_gmm_scratch, block_active_scratch, element_count, component_count, mixture_count);

        for (int i=1; i < k; ++i)
        {
            GMMReductionKernel<4, false><<<grid, block>>>(i, image, alpha, block_gmm_scratch, block_active_scratch, element_count, component_count, mixture_count);
        }

        GMMFinalizeKernel<4, false><<<k, block>>>(block_gmm_scratch, gmm, grid.x, component_count);

        GMMFindSplit<<<1, dim3(BLOCK_SIZE, mixture_count)>>>(gmm_split_scratch, k / mixture_count, gmm, component_count, mixture_count);
        GMMDoSplit<<<TILE(element_count, BLOCK_SIZE * DO_SPLIT_DEGENERACY), BLOCK_SIZE>>>(gmm_split_scratch, (k / mixture_count) << 4, gmm, component_count, image, alpha, element_count);
    }
}

void GMMUpdate(const float *image, int *alpha, float *gmm, float *scratch_mem, int element_count, int mixture_count, int mixture_size, int component_count)
{
    dim3 grid(TILE(element_count, BLOCK_SIZE * BLOCK_SIZE));
    dim3 block(BLOCK_SIZE * 4);

    float* block_gmm_scratch = &scratch_mem[grid.x];
    unsigned int* block_active_scratch = (unsigned int*)scratch_mem;

    int gmm_N = mixture_count * mixture_size;

    GMMReductionKernel<4, true><<<grid, block>>>(0, image, alpha, block_gmm_scratch, block_active_scratch, element_count, component_count, mixture_count);

    for (int i = 1; i < gmm_N; ++i)
    {
        GMMReductionKernel<4, false><<<grid, block>>>(i, image, alpha, block_gmm_scratch, block_active_scratch, element_count, component_count, mixture_count);
    }

    GMMFinalizeKernel<4, true><<<gmm_N, block>>>(block_gmm_scratch, gmm, grid.x, component_count);

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
    
    GMMInitialize(input, alpha, gmm, scratch_mem, element_count, mixture_count, mixture_size, component_count);
    GMMUpdate(input, alpha, gmm, scratch_mem, element_count, mixture_count, mixture_size, component_count);
    GMMDataTerm(input, gmm, output, element_count, mixture_count, mixture_size, component_count);

    cudaFree(alpha);
    cudaFree(gmm);
}
