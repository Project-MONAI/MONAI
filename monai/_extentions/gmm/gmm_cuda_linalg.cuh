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

#if CHANNEL_COUNT == 1

    #define DET_FACTORS 1
    #define DET_TERMS 1
    #define DET_REDUCTIONS 0
    #define DET_SHFL_MASK 0x0
    #define DET_SHFL_WIDTH 0

    __constant__ int c_det_indices[2] {
        1,
        0,
    };

    #define INV_COMPONENTS 1
    #define INV_TERMS 1
    #define INV_FACTORS 0

    __constant__ int c_inv_indices[1] {
        1
    };

#elif CHANNEL_COUNT == 2

    #define DET_FACTORS 2
    #define DET_TERMS 2
    #define DET_REDUCTIONS 1
    #define DET_SHFL_MASK 0x3
    #define DET_SHFL_WIDTH 2

    __constant__ int c_det_indices[6] {
        1, -1,
        0,  1,
        2,  1,
    };

    #define INV_COMPONENTS 3
    #define INV_TERMS 1
    #define INV_FACTORS 1

    __constant__ int c_inv_indices[6] {
        1, -1,  1,
        2,  1,  0,
    };

#elif CHANNEL_COUNT == 3

    #define DET_FACTORS 3
    #define DET_TERMS 5
    #define DET_REDUCTIONS 3
    #define DET_SHFL_MASK 0x1f
    #define DET_SHFL_WIDTH 8

    __constant__ int c_det_indices[20] {
        1, -1, -1,  2, -1,
        0,  0,  1,  1,  2,
        3,  4,  1,  2,  2,
        5,  4,  5,  4,  3,
    };

    #define INV_COMPONENTS 6
    #define INV_TERMS 2
    #define INV_FACTORS 2

    __constant__ int c_inv_indices[36] {
        1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, 
        3,  4,  2,  1,  1,  2,  0,  2,  1,  0,  0,  1, 
        5,  4,  4,  5,  4,  3,  5,  2,  2,  4,  3,  1, 
    };

#elif CHANNEL_COUNT == 4

    #define DET_FACTORS 4
    #define DET_TERMS 17
    #define DET_REDUCTIONS 5
    #define DET_SHFL_MASK 0x1ffff
    #define DET_SHFL_WIDTH 32

    __constant__ int c_det_indices[85] {
        1, -1, -1,  2, -1, -1,  1,  2, -2, -2,  2, -1,  1,  2, -2, -1,  1,
        0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3, 
        4,  4,  5,  5,  6,  1,  1,  2,  2,  3,  3,  2,  2,  3,  3,  3,  3, 
        7,  8,  5,  6,  6,  7,  8,  5,  6,  5,  6,  4,  6,  4,  5,  4,  5, 
        9,  8,  9,  8,  7,  9,  8,  9,  8,  8,  7,  9,  6,  8,  6,  7,  5, 
    };
    
    #define INV_COMPONENTS 10
    #define INV_TERMS 6
    #define INV_FACTORS 3

    __constant__ int c_inv_indices[240] {
        1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1,
        4,  4,  5,  5,  5,  6,  1,  1,  2,  2,  3,  3,  1,  1,  2,  2,  3,  3,  1,  1,  2,  2,  3,  3,  0,  0,  2,  2,  2,  3,  0,  0,  1,  1,  2,  3,  0,  0,  1,  1,  2,  2,  0,  0,  1,  1,  1,  3,  0,  0,  1,  1,  1,  2,  0,  0,  1,  1,  1,  2,
        7,  8,  5,  6,  6,  6,  7,  8,  5,  6,  5,  6,  5,  6,  4,  6,  4,  5,  5,  6,  4,  5,  4,  5,  7,  8,  2,  3,  3,  3,  5,  6,  2,  3,  3,  3,  5,  6,  2,  3,  2,  3,  4,  6,  1,  3,  3,  3,  4,  5,  1,  2,  3,  3,  4,  5,  1,  2,  2,  2,
        9,  8,  9,  8,  8,  7,  9,  8,  9,  8,  8,  7,  9,  8,  9,  6,  8,  6,  8,  7,  8,  6,  7,  5,  9,  8,  9,  8,  8,  7,  9,  8,  9,  8,  6,  5,  8,  7,  8,  7,  6,  5,  9,  6,  9,  6,  6,  4,  8,  6,  8,  6,  5,  4,  7,  5,  7,  5,  5,  4,
    };

#endif 

__device__ __forceinline__ void CalculateDeterminant(float* matrix, float* determinant, int thread_index)
{    
    if (thread_index < DET_TERMS)
    {
        float det_term = c_det_indices[thread_index];

        for (int i = 0; i < DET_FACTORS; i++)
        {
            int index = c_det_indices[thread_index + (i + 1) * DET_TERMS];
            det_term *= matrix[index];
        }

        for (int i = DET_REDUCTIONS - 1; i >= 0; i--)
        {
            det_term += __shfl_down_sync(DET_SHFL_MASK, det_term, 1 << i, DET_SHFL_WIDTH);
        }

        if(thread_index == 0)
        {
            *determinant = det_term;
        }
    }
}

__device__ __forceinline__ void InvertMatrix(float* matrix, float determinant, int thread_index)
{
    if (thread_index < INV_COMPONENTS)
    {
        if (determinant > 0.0f)
        {
            float inverse_element = 0.0f;

            for (int i = 0; i < INV_TERMS; i++)
            {
                float term = c_inv_indices[thread_index * INV_TERMS + i + 0 * INV_TERMS * INV_COMPONENTS];

                for (int j = 0; j < INV_FACTORS; j++)
                {
                    int index = c_inv_indices[thread_index * INV_TERMS + i + (j+1) * INV_TERMS * INV_COMPONENTS];
                    term *= matrix[index];
                }

                inverse_element += term;
            }

            matrix[thread_index] = inverse_element / determinant;
        }
        else
        {
            matrix[thread_index] = 0.0f;
        }
    }
}

__device__ void normalize(float* v)
{
    float norm = 0.0f;

    for (int i = 0; i < CHANNEL_COUNT; i++)
    {
        norm += v[i] * v[i];
    }

    norm = 1.0f / sqrtf(norm);

    for (int i = 0; i < CHANNEL_COUNT; i++)
    {
        v[i] *= norm;
    }
}

__device__ float scalar_prod(float* a, float* b)
{
    float product = 0.0f;

    for (int i = 0; i < CHANNEL_COUNT; i++)
    {
        product += a[i] * b[i];
    }

    return product;
}

__device__ void largest_eigenpair(const float *M, float* evec, float* eval)
{
    float scratch[CHANNEL_COUNT];

    for(int i = 0; i < CHANNEL_COUNT; i++)
    {
        scratch[i] = i;
    }

    for (int itr = 0; itr < 10; itr++)
    {
        *eval = 0.0f;

        for (int i = 0; i < CHANNEL_COUNT; i++)
        {
            int index = i;

            for (int j = 0; j < CHANNEL_COUNT; j++)
            {
                evec[i] += M[index] * scratch[j];

                if (j < i)
                {
                    index += CHANNEL_COUNT - (j + 1);
                }
                else
                {
                    index += 1;
                }
            }

            *eval += evec[i] * evec[i];
        }

        *eval = sqrtf(*eval);

        for (int i = 0; i < CHANNEL_COUNT; i++)
        {
            evec[i] /= *eval;
            scratch[i] = evec[i];
        }
    }
}
