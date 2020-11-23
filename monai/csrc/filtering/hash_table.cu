#include "cuda_memory.h"

//#define USE_ADDITIVE_HASH

// turn this on if you want to get slighly less memory consumption and slightly longer run times.
//#define LINEAR_D_MEMORY 

#define USE_CUSTOM_MODULO

__device__ __constant__ float *table_values;
__device__ __constant__ signed short *table_keys;
__device__ __constant__ int *table_entries;
__device__ __constant__ unsigned int table_capacity;
__device__ __constant__ signed short *table_zeros;
__device__ __constant__ char *table_rank;





/*************************************************************/
/* Fast computation of modulo operator with constant divisor */
/*************************************************************/
__device__ __constant__ unsigned int __div_m;
__device__ __constant__ unsigned int __div_l;
__device__ __constant__ unsigned int __div_c;

#ifdef USE_CUSTOM_MODULO
__device__ inline unsigned int modHash(unsigned int n) {
  unsigned int t1 = __umulhi(__div_m, n);
  return n - ((t1+((n-t1)>>1))>>(__div_l-1)) * __div_c;
}

#else
#define modHash(n) ((n)%(2*table_capacity));
#endif

/*************************************************************/
/* End modulo                                                */
/*************************************************************/

__device__ __constant__ unsigned int hOffset[64];

template<int kd, int vd>
void createHashTable(int capacity) {

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_capacity,
				      &capacity,
				      sizeof(unsigned int)));
        
    float *values;
    allocateCudaMemory((void**)&values, capacity*vd*sizeof(float));
    CUDA_SAFE_CALL(cudaMemset((void *)values, 0, capacity*vd*sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_values,
				      &values,
				      sizeof(float *)));       
    

    int *entries;
    allocateCudaMemory((void **)&entries, capacity*2*sizeof(int));
    CUDA_SAFE_CALL(cudaMemset((void *)entries, -1, capacity*2*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_entries, 
				      &entries,
				      sizeof(unsigned int *)));

    #ifdef LINEAR_D_MEMORY
    char *ranks;
    allocateCudaMemory((void**)&ranks, capacity*sizeof(char));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_rank,
				      &ranks,
				      sizeof(char *)));

    signed short *zeros;
    allocateCudaMemory((void**)&zeros, capacity*sizeof(signed short));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_zeros,
				      &zeros,
				      sizeof(char *)));

    #else

    signed short *keys;
    allocateCudaMemory((void **)&keys, capacity*kd*sizeof(signed short));
    CUDA_SAFE_CALL(cudaMemset((void *)keys, 0, capacity*kd*sizeof(signed short)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_keys,
				      &keys,
				      sizeof(unsigned int *)));
    #endif

}

template <int vd> static void resetHashTable() {
  CUDA_SAFE_CALL(cudaMemset((void*)table_values, 0, table_capacity*vd*sizeof(float)));
}

static void destroyHashTable() {
#ifndef LINEAR_D_MEMORY
    CUDA_SAFE_CALL(cudaFree(table_keys));
#endif
    CUDA_SAFE_CALL(cudaFree(table_values));
    CUDA_SAFE_CALL(cudaFree(table_entries));
}

template<int kd> __device__ __host__ static unsigned int hash(signed short *key) {
    unsigned int k = 0; 
    for (int i = 0; i < kd; i++) {
	k += key[i];
	k = k * 2531011; 
    }
    return k;
}

template<int kd> __device__ __host__ static unsigned int hash(int *key) {
    unsigned int k = 0; 
    for (int i = 0; i < kd; i++) {
	k += key[i];
	k = k * 2531011; 
    }
    return k;
}

template<int d> __device__ static bool matchKey(int idx, signed short * key) {
    bool match = true;
    int slot = idx/(d+1), color = idx-slot*(d+1);
    char *rank = table_rank + slot * (d+1);
    signed short *zero = table_zeros + slot * (d+1);
    for (int i = 0; i < d && match; i++) {
	match = (key[i] == zero[i] + color - (rank[i] > d-color ? (d+1) : 0));
    }
    return match;
}


template<int d> __device__ static void generateKey(int idx, signed short * key) {
    int slot = idx/(d+1), color = idx-slot*(d+1);
    char *rank = table_rank + slot * (d+1);
    signed short *zero = table_zeros + slot * (d+1);
    for (int i = 0; i < d; i++) {
	key[i] = zero[i] + color - (rank[i] > d-color ? (d+1) : 0);
    }
}

static float* swapHashTableValues(float *newValues) {
    float * oldValues;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&oldValues,
					table_values,
					sizeof(float *)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(table_values,
				      &newValues,
				      sizeof(float *)));
    return oldValues;
}


template<int kd>
__device__ static int hashTableInsert(unsigned int fh, signed short *key, unsigned int slot) {	
    int h = modHash(fh);
    while (1) {
	int *e = &table_entries[h];

	// If the cell is empty (-1), lock it (-2)
	int contents = atomicCAS(e, -1, -2);

	if (contents == -2) {
	    // If it was locked already, move on to the next cell

	} else if (contents == -1) { 
	    // If it was empty, we successfully locked it. Write our key.

	    #ifndef LINEAR_D_MEMORY
	    for (int i = 0; i < kd; i++) {
		table_keys[slot*kd+i] = key[i];
	    }
	    #endif

	    // Unlock
	    atomicExch(e, slot); 

	    return h;
	} else {
	    // The cell is unlocked and has a key in it, check if it matches
            #ifdef LINEAR_D_MEMORY
 	    if (matchKey<kd>(contents, key)) return h;
            #else
	    bool match = true;
	    for (int i = 0; i < kd && match; i++) {
		match = (table_keys[contents*kd+i] == key[i]);
	    }
	    if (match) return h;
            #endif       

	}
	// increment the bucket with wraparound
	h++;
	if (h == table_capacity*2) h = 0;
    }
}

template<int kd>
__device__ static int hashTableInsert(signed short *key, unsigned int slot) {
    unsigned int myHash = hash<kd>(key);
    return hashTableInsert<kd>(myHash, key, slot);
}



template<int kd> __device__ static
int hashTableRetrieveWithHash(unsigned int fh, signed short *key) {
  int h = modHash(fh);
  while (1) {
    int *e = table_entries + h;
    
    if (*e == -1) return -1;
    
    #ifdef LINEAR_D_MEMORY
    if (matchKey<kd>((*e), key)) return *e;
    #else
    bool match = true;
    for (int i = 0; i < kd && match; i++) {
      match = (table_keys[(*e)*kd+i] == key[i]);
    }
    if (match) return *e;
    #endif
    
    h++;
    if (h == table_capacity*2) h = 0;
  }
}
   
template<int kd>
__device__ static int hashTableRetrieve(signed short *key) {

    int h = modHash(hash<kd>(key));
    while (1) {
	int *e = table_entries + h;

	if (*e == -1) return -1;

	#ifdef LINEAR_D_MEMORY
	if (matchKey<kd>((*e), key)) return *e;
	#else
	bool match = true;
	for (int i = 0; i < kd && match; i++) {
	    match = (table_keys[(*e)*kd+i] == key[i]);
	}
	if (match) return *e;
	#endif

	h++;
	if (h == table_capacity*2) h = 0;
    }
}








