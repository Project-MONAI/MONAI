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

__device__ void to_square(float in[SUB_MATRIX_COMPONENT_COUNT], float out[CHANNEL_COUNT][CHANNEL_COUNT])
{
    for (int index = 0, i = 0; i < CHANNEL_COUNT; i++)
    {
        for (int j = i; j < CHANNEL_COUNT; j++, index++)
        {
            out[i][j] = in[index];
            out[j][i] = in[index];
        }
    }
}

__device__ void to_triangle(float in[CHANNEL_COUNT][CHANNEL_COUNT], float out[SUB_MATRIX_COMPONENT_COUNT])
{
    for (int index = 0, i = 0; i < CHANNEL_COUNT; i++)
    {
        for (int j = i; j < CHANNEL_COUNT; j++, index++)
        {
            out[index] = in[j][i];
        }
    }
}

__device__ void cholesky(float in[CHANNEL_COUNT][CHANNEL_COUNT], float out[CHANNEL_COUNT][CHANNEL_COUNT])
{
    for (int i = 0; i < CHANNEL_COUNT; i++)
    {
        for (int j = 0; j < i+1; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < j; k++)
            {
                sum += out[i][k] * out[j][k];
            }

            if (i == j)
            {
                out[i][j] = sqrtf(in[i][i] - sum);
            }
            else
            {
                out[i][j] = (in[i][j] - sum) / out[j][j];
            }
        }
    }
}

__device__ float chol_det(float in[CHANNEL_COUNT][CHANNEL_COUNT])
{
    float det = 1.0f;

    for (int i = 0; i < CHANNEL_COUNT; i++)
    {
        det *= in[i][i];
    }

    return det * det;
}

__device__ void chol_inv(float in[CHANNEL_COUNT][CHANNEL_COUNT], float out[CHANNEL_COUNT][CHANNEL_COUNT])
{
    // Invert cholesky matrix
    for (int i = 0; i < CHANNEL_COUNT; i++)
    {
        in[i][i] = 1.0f / (in[i][i] + 0.0001f);

        for (int j = 0; j < i; j++)
        {
            float sum = 0.0f;

            for (int k = j; k < i; k++)
            {
                sum += in[i][k] * in[k][j];
            }

            in[i][j] = -in[i][i] * sum;
        }
    }

    // Dot with transpose of self
    for (int i = 0; i < CHANNEL_COUNT; i++)
    {
        for (int j = 0; j < CHANNEL_COUNT; j++)
        {
            out[i][j] = 0.0f;

            for (int k = max(i, j); k < CHANNEL_COUNT; k++)
            {
                out[i][j] += in[k][i] * in[k][j];
            }
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
        scratch[i] = i + 1;
    }

    for (int itr = 0; itr < 10; itr++)
    {
        *eval = 0.0f;

        for (int i = 0; i < CHANNEL_COUNT; i++)
        {
            int index = i;

            evec[i] = 0.0f;

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

            *eval = max(*eval, evec[i]);
        }

        for (int i = 0; i < CHANNEL_COUNT; i++)
        {
            evec[i] /= *eval;
            scratch[i] = evec[i];
        }
    }
}
