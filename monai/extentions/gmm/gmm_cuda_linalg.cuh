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
    #define DET_SHFL_MASK 0x1
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

    //to do
    __constant__ int c_inv_indices[180] {
        1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, 
        3,  4,  2,  1,  1,  2,  0,  2,  1,  0,  0,  1, 
        5,  4,  4,  5,  4,  3,  5,  2,  2,  4,  3,  1, 
    };

// +ehJ -eoo -ffJ +2fgo -ggh | -bhJ +boo +cfJ -cgo -dfo +dgh    | bfJ -bgo -ceJ +cgg +deo -dfg  | -bfo +bgh +ceo -cfg -deh +dff
//                           | ahJ -aoo -ccJ +2cd0 -ddh         | -afJ +ago +bcj -bdo -cdg +ddf | afo -agh -bco +bdh +ccg -cdf
//                                                              | aeJ -agg -bbJ +2bdg -dde      | -aeo + afg +bbo -bcg -bdf +cde
//                                                                                              | aeh -aff -bbh +2bcf -cce

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
