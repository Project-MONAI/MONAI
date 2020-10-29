/*
Copyright 2020 MONAI Consortium
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
#include <torch/extension.h>

#include <vector>
#include <cstdio>
#include <utility>



void cudaErrorCheck(int x){
    auto code = cudaGetLastError();
    if(cudaSuccess != code){
        fprintf(stderr,"GPU Error %d: %s\n", x, cudaGetErrorString(code));
        exit(code);
    }
}


__global__ void insert_kernel(const torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> keys,
                              const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> hashs,
                              torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> table,
                              torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> n_entries,
                              torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> actual_hash,
                              const unsigned int pos_dim,
                              const unsigned int n,
                              const unsigned int capacity) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n){
        unsigned int table_idx = hashs[idx];
        bool need_to_check = false;
        for (int l=0; l<10000; l++){
            /// table_idx is the line in the table gaved by the hash function
            /// If the cell is empty (<0), lock it (0) (returns -1 or -2)
            /// If the cell is locked (0), keep it locked (0) (returns 0)
            /// If the cell contains content (#number>0), keep it like this (returns #number)
            // int contents = atomicCAS(&(table[table_idx][pos_dim]), -1, -2);
            int contents = atomicMax(&(table[table_idx][pos_dim]), 0);

            /// If it encounter a locks cell, we will have to check later for unicity.
            if (contents == 0){
                need_to_check = true;
            }
            /// If it was empty, we successfully locked it. Write our key.
            else if (contents<0){
                /// Check how many entries in the table already (#n_curr) and add 1.
                int n_curr = atomicAdd(&(n_entries[0]), 1);
                /// Write the key itself
                for (int i = 0; i < pos_dim; i++){
                    table[table_idx][i] = keys[idx][i];
                }
                /// Unlock the block and write the corresponding key number instead.
                atomicExch(&(table[table_idx][pos_dim]), n_curr);
                /// In case a check is needed, mark it as need to check 
                if (need_to_check){
                    actual_hash[idx][0] = table_idx;
                    actual_hash[idx][1] = 1;
                }
                break;
            /// Otherwise we check if the cell corresponds to our query.
            } else {
                /// The cell contains a non default block, check if it matches
                bool match = true;
                for (int i = 0; i < pos_dim && match; i++){
                    match = (table[table_idx][i] == keys[idx][i]);
                }
                if (match){
                    if (need_to_check){
                        actual_hash[idx][0] = table_idx;
                        actual_hash[idx][1] = 1;
                    }
                    break;
                }
            }
            // increment the bucket with wraparound
            table_idx++;
            if (table_idx == capacity)
                table_idx = 0;
        }
    }
}

__global__ void clean_table(torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> table,
                            const torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> keys,
                            const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> hashs,
                            const torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> actual_hash,
                            int pos_dim,
                            int n,
                            int capacity){
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n){ 
        if (actual_hash[idx][1]==1){
            unsigned int table_idx = hashs[idx];
            for (int l=0; l<10000; l++){
                /// We start searching at the hash loc
                int contents = table[table_idx][pos_dim];
                if (contents == -2){
                    /// The key was not incerted, we reached the end of possible values
                    /// In practice we should never end up in that scope because actual_hash is written only if the key appears in the table (written or found)
                    break;
                }
                else if (contents==-1) {
                    /// The content of this cell has been deleted because it was duplicated, need to keep searching
                }
                else {
                    /// Check that the key are matching, then compare actual_hash with current hash
                    bool match = true;
                    for (int i = 0; i < pos_dim && match; i++){
                        match = (table[table_idx][i] == keys[idx][i]);
                    }
                    if (match){
                        int del_idx = actual_hash[idx][0];
                        if (table_idx!=del_idx){
                            for (int i = 0; i < pos_dim; i++){
                                table[del_idx][i] = -1;
                            }
                            table[del_idx][pos_dim] = -1;
                        }
                        break;
                    }
                }
                // increment the bucket with wraparound
                table_idx++;
                if (table_idx == capacity)
                    table_idx = 0;
            }
        }
    }
}

__global__ void renumber(torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> table,
                         torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> n_entries,
                         int capacity,
                         int pos_dim){
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx<capacity){
        if (table[idx][pos_dim]>0){
            int new_rk = atomicAdd(&(n_entries[0]), 1);
            table[idx][pos_dim] = new_rk;
        }
    }
}

__host__ std::vector<torch::Tensor> insert_cuda(torch::Tensor table,
                                                torch::Tensor n_entries,
                                                torch::Tensor keys,
                                                torch::Tensor hashs){
    /// table: capacity , (pos_dim + 1)
    /// n_entries: 1,     !!!!!!! It has to be initialized as [1] !!!!!!!
    /// keys: number_of_new_entries , pos_dim
    /// hashs: number_of_new_entries,
    unsigned int n = hashs.size(0);
    unsigned int pos_dim = keys.size(1);
    unsigned int capacity = table.size(0);
    torch::Tensor actual_hash = torch::zeros({n, 2}, hashs.options());
    const int threads = 64;
    const dim3 blocks((n + threads - 1) / threads, 1, 1);
    insert_kernel <<<blocks, threads>>> (keys.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                         hashs.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                                         table.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                         n_entries.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                                         actual_hash.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                         pos_dim,
                                         n,
                                         capacity);
    cudaDeviceSynchronize();
    clean_table <<<blocks, threads>>> (table.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                       keys.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                       hashs.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                                       actual_hash.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                       pos_dim,
                                       n,
                                       capacity);
    cudaDeviceSynchronize();
    n_entries[0] = 1;
    const int threads_renumber = 64;
    const dim3 blocks_renumber((capacity + threads_renumber - 1) / threads_renumber, 1, 1);
    renumber <<<blocks_renumber, threads_renumber>>> (table.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                             n_entries.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                                             capacity,
                                             pos_dim);
    cudaDeviceSynchronize();
    return {table, actual_hash};
}


__global__ void get_rank_kernel(const torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> table,
                                const torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> keys,
                                const torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> hashs,
                                torch::PackedTensorAccessor<int, 1, torch::RestrictPtrTraits, size_t> rank,
                                int pos_dim,
                                int n,
                                int capacity){
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx<n){
        unsigned int table_idx = hashs[idx];
        for (int l=0; l<10000; l++){
            /// We start searching at the hash loc
            int contents = table[table_idx][pos_dim];
            if (contents == -2){
                /// The key was not incerted, we reached the end of possible values
                break;
            }
            else if (contents==-1) {
                /// The content of this cell has been deleted because it was duplicated, need to keep searching
            }
            else {
                /// Check that the key are matching
                bool match = true;
                for (int i = 0; i < pos_dim && match; i++){
                    match = (table[table_idx][i] == keys[idx][i]);
                }
                if (match){
                    rank[idx] = contents;
                    break;
                }
            }
            // increment the bucket with wraparound
            table_idx++;
            if (table_idx == capacity)
                table_idx = 0;
        }
    }                          
}


__host__ torch::Tensor get_rank_cuda(torch::Tensor table,
                                     torch::Tensor keys,
                                     torch::Tensor hash){
    /// table: capacity , (pos_dim + 1)
    /// keys: number_of_new_entries , pos_dim
    /// hashs: number_of_new_entries,
    unsigned int n = hash.size(0);
    unsigned int pos_dim = keys.size(1);
    unsigned int capacity = table.size(0);
    torch::Tensor rank = torch::zeros({n}, hash.options());
    const int threads = 64;
    const dim3 blocks((n + threads - 1) / threads, 1, 1);
    get_rank_kernel <<<blocks, threads>>> (table.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                           keys.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                           hash.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                                           rank.packed_accessor<int, 1, torch::RestrictPtrTraits, size_t>(),
                                           pos_dim, 
                                           n,
                                           capacity);
    return rank;
}


__global__ void get_values_kernel(torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> value_table,
                                  const torch::PackedTensorAccessor<int, 2, torch::RestrictPtrTraits, size_t> table,
                                  int n,
                                  int pos_dim){
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx<n){
        if (table[idx][pos_dim]>0){
            int table_idx = table[idx][pos_dim];
            for (int i=0; i<pos_dim; i++){
                value_table[table_idx-1][i] = table[idx][i];
            }
        }
    }
}


__host__ torch::Tensor get_values_cuda(torch::Tensor table,
                                       int n_entries){
    const int table_size = table.size(0);
    const int pos_dim = table.size(1) - 1;
    torch::Tensor value_table = torch::zeros({n_entries-1, pos_dim}, table.options());
    const int threads = 64;
    const dim3 blocks((table_size + threads - 1) / threads, 1, 1);
    get_values_kernel <<<blocks, threads>>> (value_table.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                             table.packed_accessor<int, 2, torch::RestrictPtrTraits, size_t>(),
                                             table_size,
                                             pos_dim);
    cudaDeviceSynchronize();
    return value_table;
}