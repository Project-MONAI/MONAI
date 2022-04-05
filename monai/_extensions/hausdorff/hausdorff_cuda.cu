#include <torch/extension.h>
//#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
#include <cuda/pipeline>
#include <vector>
//#include <cuda/annotated_ptr>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>

#include <cmath>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <assert.h>
#include "device_launch_parameters.h"

using namespace cooperative_groups;


#pragma once

//constants describing the meaning of main shared memory spaces
constexpr uint32_t localWorkQueLength = 300;
constexpr uint32_t startOfLocalWorkQ = 4160;
constexpr uint32_t lengthOfMainShmem = 4460;//4460;
constexpr uint32_t begResShmem = 1088;
constexpr uint32_t begfirstRegShmem = 2112;
constexpr uint32_t begSecRegShmem = 3136;
constexpr uint32_t begSMallRegShmemA = 0;
constexpr uint32_t begSMallRegShmemB = 1056;
constexpr uint32_t begSourceShmem = 32;

//as the pipeline is so asynchronous we need some additional storage for saving data related to the operations
//will define how big is the  amount of space in order to be able to use fast version of modulo operator it should be power of 2 ...
constexpr uint32_t modForPipelineVars = 16;
constexpr uint32_t fpLocCountBeg = 4460;




// other
//added to linear index meta in order to  mark weather block is of type gold or not
constexpr uint32_t  isGoldOffset = (UINT16_MAX * 10);


/**
In order to be able to use cuda malloc 3d we will implemnt it as a series
of 3d arrays
*/


#pragma once
extern "C" struct MetaDataCPU {
    int metaXLength;
    int MetaYLength;
    int MetaZLength;
    int totalMetaLength;


    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ - minimal and maximum coordinates of blocks with some entries of intrest
    //7)global FP count; 8)global FN count  9) workQueueCounter 10)resultFP globalCounter 11) resultFn globalCounter
     //12) global FPandFn offset 13)globalIterationNumb
    //array3dWithDimsCPU<unsigned int> minMaxes;
    unsigned int* minMaxes;

    ////// counts of false positive and false negatives in given metadata blocks

    ///// sizes of array below will be established on the basis of fp and fn values known after boolKernel finished execution

    //work queue -  workqueue counter already present in minMaxes as entry 9
    //in practice it is matrix of length the same as FP+FN global count +1 and width of 5
         //1) xMeta; 2)yMeta 3)zMeta 4)isGold 5)iteration number
    //we use one single long rewsult list - in order to avoid overwriting each block each block has established offset where it would write it's results
    uint32_t* resultList;

};

#pragma once
extern "C" struct MetaDataGPU {
    int metaXLength;
    int MetaYLength;
    int MetaZLength;
    int totalMetaLength;

    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ - minimal and maximum coordinates of blocks with some entries of intrest
    //7)global FP count; 8)global FN count 9) workQueueCounter 10)resultFP globalCounter 11) resultFn globalCounter
    //12) global FPandFn offset 13)globalIterationNumb

    unsigned int* minMaxes;

    //represents x from description of main Arr
    unsigned int mainArrXLength;
    //have length 4x
    unsigned int mainArrSectionLength;

    unsigned int metaDataSectionLength = 20;

    // now we will store here also calculated by min maxes kernel values of minimum and maximumvalues
        //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    unsigned int maxX;
    unsigned int minX;
    unsigned int maxY;
    unsigned int minY;
    unsigned int maxZ;
    unsigned int minZ;
};








/*
* Basically holding the arguments for master functions controlling full preparation to get all for Housedorff kernel
*/
#pragma once
template <typename TFF>
struct ForFullBoolPrepArgs {

    //metadata struct
    MetaDataCPU metaData;

    // dimensions of data block
    int dbXLength;
    int dbYLength;
    int dbZLength;
    // gold standard and segmentation output array
    torch::Tensor goldArr;
    torch::Tensor segmArr;

    int Nx;
    int Ny;
    int Nz;

    TFF numberToLookFor;// what we will look for in arrays
    //number and dimensionality of threads and blocks required to lounch bool kernel
    dim3 threads;
    int blocks;
    //threads and blocks for first metadata pass kernel
    int threadsFirstMetaDataPass;
    int blocksFirstMetaDataPass;
    //threads and blocks for main pass
    dim3 threadsMainPass;
    int blocksMainPass;
    //threads and blocks for padding pass
    dim3 threadsPaddingPass;
    int blocksPaddingPass;
    //threads and blocks for non first metadata passes
    int threadsOtherMetaDataPasses;
    int blocksOtherMetaDataPasses;
    // will establish how many points we want to include in dilatation and how many we can ignore so typically set to 95% - so we will ignore only 5% most distant
    float robustnessPercent = 1.0;  // 0.95;

    int32_t* resultListPointerMeta;
    int32_t* resultListPointerLocalCPU;
    int32_t* resultListPointerIterNumb;

};


/*
* Basically holding the arguments for main kernel in the FullBoolPrep
*/
#pragma once
template <typename TFB>
struct ForBoolKernelArgs {
    //matadata struct
    MetaDataGPU metaData;
    // dimensions of data block
    int dbXLength;
    int dbYLength;
    int dbZLength;
    // gold standard and segmentation output array
    torch::Tensor goldArr;
    torch::Tensor segmArr;
    TFB numberToLookFor;
    int Nx;
    int Ny;
    int Nz;


    // Frequent accesses to "a" and "b"; infrequent accesses to "x" and "y":
    //cuda::annotated_ptr<int const, cuda::access_property::persisting> a_p{ a }, b_p{ b };
    //cuda::annotated_ptr<int, cuda::access_property::streaming> x_s{ x }, y_s{ y };

    int32_t* resultListPointerMeta;
    int32_t* resultListPointerLocal;
    int32_t* resultListPointerIterNumb;

    torch::Tensor resultListPointerIterNumbTensor;




    uint32_t* origArrsPointer;
    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    uint32_t* metaDataArrPointer;

    uint32_t* workQueuePointer;
    unsigned int* minMaxes;


    /*
main array with all required data  organized in sections for each metadata block
x-  is block dimx times block dimy
now what occupies what positions
##mainArrA
(0) - x-1 : reducedGoldRef
(x) - 2x-1 : reducedSegmRef
##mainArrB
() - 3x-1 : reducedGoldPrev
(x) - 4x-1 : reducedSegmPrev

##metaDataArr
0: empty
1 :fpCount
2 :fnCount
3 :fpCounter
4 :fnCounter
5 :fpOffset
6 :fnOffset
7 :isActiveGold
8 :isFullGold
9 :isActiveSegm
10 :isFullSegm
11 :isToBeActivatedGold
12 :isToBeActivatedSegm
//now linear indexes of the blocks in all sides - if there is no block in given direction it will equal UINT32_MAX
13 : top
14 : bottom
15 : left
16 : right
17 : anterior
18 : posterior
19 : empty
20 : empty


###origArrs
0-x : reducedGold
(x+1) - 2x : reducedSegm

*/
//   uint32_t* mainArr;




    float robustnessPercent = 1.0;// 0.95;

};





//just utility for unit testing - set some data bout points
#pragma once
extern "C"  struct forTestPointStruct {
    int x;
    int y;
    int z;

    bool isGold;
    bool isGoldAndSegm;

    int xMeta;
    int yMeta;
    int zMeta;



    bool shouldBeInResAfterOneDil;
    bool shouldBeInResAfterTwoDil;
    bool isFoundAndDilatated = false;
    bool isFoundInResult = false;

    bool isFoundDilTop = false;
    bool isFoundDilBottom = false;

    bool isFoundDilAnterior = false;
    bool isFoundDilPosterior = false;

    bool isFoundDilLeft = false;
    bool isFoundDilRight = false;

};


#pragma once
extern "C" struct forTestMetaDataStruct {

    int xMeta;
    int yMeta;
    int zMeta;

    int requiredspaceInFpResultList;
    int requiredspaceInFnResultList;

    bool isToBeActiveAtStart;
    bool isToBeActiveAfterOneIter;
    bool isToBeActiveAfterTwoIter;

    bool isToBeValidatedFpAfterOneIter;
    bool isToBeValidatedFpAfterTwoIter;

    bool isToBeValidatedFnAfterOneIter;
    bool isToBeValidatedFnAfterTwoIter;


    bool isToBeFullAfterOneIter;
    bool isToBeFullAfterTwoIter;

    int fpCount;
    int fnCount;

    int fpConterAfterOneDil;
    int fpConterAfterTwoDil;

    int fnConterAfterOneDil;
    int fnConterAfterTwoDil;


};


/*
data from occupancy calculator API used to get optimal number of thread blocks and threads per thread block
*/
struct occupancyCalcData {
    int warpsNumbForMinMax;
    int blockSizeForMinMax;

    int warpsNumbForboolPrepareKernel;
    int blockSizeFoboolPrepareKernel;

    int theadsForFirstMetaPass;
    int blockForFirstMetaPass;

    int warpsNumbForMainPass;
    int blockForMainPass;
};



/*
copy from host to device
*/
#pragma once
inline MetaDataGPU allocateMetaDataOnGPU(MetaDataCPU metaDataCPU, unsigned int*& minMaxes) {
    MetaDataGPU res;

    metaDataCPU.minMaxes[1] = 0;
    metaDataCPU.minMaxes[2] = 1000;
    metaDataCPU.minMaxes[3] = 0;
    metaDataCPU.minMaxes[4] = 1000;
    metaDataCPU.minMaxes[5] = 0;
    metaDataCPU.minMaxes[6] = 1000;
    metaDataCPU.minMaxes[7] = 0;
    metaDataCPU.minMaxes[8] = 0;
    metaDataCPU.minMaxes[9] = 0;
    metaDataCPU.minMaxes[10] = 0;
    metaDataCPU.minMaxes[11] = 0;
    metaDataCPU.minMaxes[12] = 0;
    metaDataCPU.minMaxes[13] = 0;
    metaDataCPU.minMaxes[14] = 0;
    metaDataCPU.minMaxes[15] = 0;
    metaDataCPU.minMaxes[16] = 0;
    metaDataCPU.minMaxes[17] = 0;
    metaDataCPU.minMaxes[18] = 0;
    metaDataCPU.minMaxes[19] = 0;
    metaDataCPU.minMaxes[20] = 0;

    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(minMaxes, metaDataCPU.minMaxes, size, cudaMemcpyHostToDevice);

    //res.resultList = allocate3dInGPU(metaDataCPU.resultList);

    //res.metaXLength = metaDataCPU.metaXLength;
    //res.MetaYLength = metaDataCPU.MetaYLength;
    //res.MetaZLength = metaDataCPU.MetaZLength;

    //res.totalMetaLength = metaDataCPU.totalMetaLength;
    //allocating on GPU and copying  cpu data onto GPU

    return res;

}

/*
copy from device to host
*/
#pragma once
inline void copyMetaDataToCPU(MetaDataCPU metaDataCPU, MetaDataGPU metaDataGPU) {
    //copyDeviceToHost3d(metaDataGPU.fpCount, metaDataCPU.fpCount);
    //copyDeviceToHost3d(metaDataGPU.fnCount, metaDataCPU.fnCount);
    size_t size = sizeof(unsigned int) * 20;

    cudaMemcpy(metaDataCPU.minMaxes, metaDataGPU.minMaxes, size, cudaMemcpyDeviceToHost);
}


#pragma once
inline cudaError_t checkCuda(cudaError_t result, std::string description)
{
    if (result != cudaSuccess) {
        std::cout << description;
        std::cout << "\n";
        // printf("%d", description);
        fprintf(stderr, "CUDA Runtime Error in %d : %s\n", description, cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}





/*
setting the linear index of metadata blocks that are in given direction if there is no such (out of range) we will save it as UINT32_MAX
*/
#pragma once
template <typename TCC>
__device__ inline void setNeighbourBlocks(ForBoolKernelArgs<TCC> fbArgs
    , uint8_t idX, uint8_t inArrIndex, bool predicate, uint32_t toAdd
    , uint32_t linIdexMeta, MetaDataGPU metaData, uint32_t localBlockMetaData[20]) {

    if ((threadIdx.x == idX) && (threadIdx.y == 0)) {
        if (predicate) {


            localBlockMetaData[inArrIndex] = (linIdexMeta + toAdd);
        }
        else {
            localBlockMetaData[inArrIndex] = isGoldOffset;
        }
    };
}




/*
iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
*/
#pragma once
template <typename TYO>
__global__ void getMinMaxes(ForBoolKernelArgs<TYO> fbArgs
    , unsigned int* minMaxes
    , TYO* goldArr
    , TYO* segmArr
    , MetaDataGPU metaData
) {

    // __global__ void getMinMaxes(unsigned int* minMaxes) {
     ////////////some initializations
    thread_block cta = this_thread_block();
    //thread_block_tile<32> tile = tiled_partition<32>(cta);



    //shared memory

    __shared__ bool anyInGold[1];
    //__shared__ uint32_t reduction_s[32];
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    __shared__ unsigned int minMaxesInShmem[7];

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) { minMaxesInShmem[1] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 0)) { minMaxesInShmem[2] = 1000; };

    if ((threadIdx.x == 3) && (threadIdx.y == 0)) { minMaxesInShmem[3] = 0; };
    if ((threadIdx.x == 4) && (threadIdx.y == 0)) { minMaxesInShmem[4] = 1000; };

    if ((threadIdx.x == 5) && (threadIdx.y == 0)) { minMaxesInShmem[5] = 0; };
    if ((threadIdx.x == 6) && (threadIdx.y == 0)) { minMaxesInShmem[6] = 1000; };

    if ((threadIdx.x == 7) && (threadIdx.y == 0)) { anyInGold[1] = false; };


    if ((threadIdx.x == 1) && (threadIdx.y == 0) && (blockIdx.x == 0)) {


    //    if (blockIdx.x == 0) {
    //        printf(" dims meta in min maxes  kernel Meta X %d MetaY %d metaZ %d dbXSize %d dbYsize %d dbZsize %d minX %d minY %d minZ \n "
    //            , metaData.metaXLength, metaData.MetaYLength, metaData.MetaZLength
    //            , fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength
    //            , metaData.minX, metaData.minY, metaData.minZ
    //        );

    }

    __syncthreads();

    /////////////////////////


    //main metadata iteration
    for (auto linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        int  xMeta = linIdexMeta % metaData.metaXLength;
        int   zMeta = int(floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * metaData.MetaYLength))));
        int   yMeta = int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)));
        //iterating over data block
        for (uint8_t xLoc = threadIdx.x; xLoc < 32; xLoc += blockDim.x) {
            uint32_t x = xMeta * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = yMeta * fbArgs.dbYLength + yLoc;//absolute position
                //if (y == 0) {
                //    printf("x %d  in min maxes \n ", x);

                //}
                if (y < fbArgs.Ny && x < fbArgs.Nx) {

                    // resetting


                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = zMeta * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.Nz) {
                            //first array gold
                            //uint8_t& zLocRef = zLoc; uint8_t& yLocRef = yLoc; uint8_t& xLocRef = xLoc;

                            // setting bits
                            bool goldBool = goldArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;  // (getTensorRow<TYU>(tensorslice, fbArgs.goldArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);
                            bool segmBool = segmArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;
                            if (goldBool || segmBool) {
                                anyInGold[0] = true;
                                //printf(" \n in min maxes dims meta in min maxes   x %d y%d z%d xMeta %d yMeta %d zMeta %d  kernel Meta X %d MetaY %d metaZ %d dbXSize %d dbYsize %d dbZsize %d minX %d minY %d minZ %d linIdexMeta %d counted %d  \n "
                                //    ,x,y,z,
                                //    xMeta,yMeta,zMeta
                                //    , metaData.metaXLength, metaData.MetaYLength, metaData.MetaZLength
                                //    , fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength
                                //    , metaData.minX, metaData.minY, metaData.minZ
                                //    , linIdexMeta
                                //    , int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)))
                                //);

                            }



                        }

                    }
                }

                //  __syncthreads();
                  //waiting so shared memory will be loaded evrywhere
                  //on single thread we do last sum reduction

                  /////////////////// setting min and maxes
  //    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
                auto active = coalesced_threads();
                sync(cta);
                active.sync();

                if ((threadIdx.x == 0) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[1] = max(xMeta, minMaxesInShmem[1]); };
                if ((threadIdx.x == 1) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[2] = min(xMeta, minMaxesInShmem[2]); };

                if ((threadIdx.x == 2) && (threadIdx.y == 0) && anyInGold[0]) {

                    minMaxesInShmem[3] = max(yMeta, minMaxesInShmem[3]);

                    //if (minMaxesInShmem[3] > 0) {
                    //    printf(" prim minMaxesInShmem maxY %d meta %d \n ", minMaxesInShmem[3], yMeta);
                    //}

                };
                if ((threadIdx.x == 3) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[4] = min(yMeta, minMaxesInShmem[4]); };

                if ((threadIdx.x == 4) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[5] = max(zMeta, minMaxesInShmem[5]); };
                if ((threadIdx.x == 5) && (threadIdx.y == 0) && anyInGold[0]) {
                    minMaxesInShmem[6] = min(zMeta, minMaxesInShmem[6]);
                    // printf("local fifth %d  \n", minMaxesInShmem[6]);
                };
                // active.sync();
                sync(cta); // just to reduce the warp divergence
                anyInGold[0] = false;




            }
        }

    }
    sync(cta);

    auto active = coalesced_threads();

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        //  printf("\n in minMaxes internal  %d \n", minMaxesInShmem[1]);
       //getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, fbArgs.metaData.minMaxes.Ny, 0, 0)[0] = 61;
        atomicMax(&minMaxes[1], minMaxesInShmem[1]);
        //atomicMax(&minMaxes[1], 2);
       // minMaxes[1] = 0;
    };

    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {

        atomicMin(&minMaxes[2], minMaxesInShmem[2]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicMax(&minMaxes[3], minMaxesInShmem[3]);
        //  printf(" minMaxesInShmem maxY %d \n ", minMaxes[3]);

    };

    if ((threadIdx.x == 2) && (threadIdx.y == 0)) {
        atomicMin(&minMaxes[4], minMaxesInShmem[4]);
        //   printf(" minMaxesInShmem minY %d \n ", minMaxes[4]);

    };



    if (threadIdx.x == 3 && threadIdx.y == 0) {
        atomicMax(&minMaxes[5], minMaxesInShmem[5]);
        //  printf(" minMaxesInShmem  %d \n ", minMaxes[5]);
    };

    if (threadIdx.x == 4 && threadIdx.y == 0) {
        atomicMin(&minMaxes[6], minMaxesInShmem[6]);
        // printf(" minMaxesInShmem  %d \n ", minMaxes[6]);

    };





}



/*
iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
*/
#pragma once
template <typename TYO>
__global__ void boolPrepareKernel(ForBoolKernelArgs<TYO> fbArgs
    , MetaDataGPU metaData, uint32_t* origArrs, uint32_t* metaDataArr
    , TYO* goldArr
    , TYO* segmArr
    , unsigned int* minMaxes) {

    ////////////some initializations
    bool goldBool = false;
    bool segmBool = false;
    bool isNotEmpty = false;

    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
    uint32_t sumFp = 0;
    uint32_t sumFn = 0;

    //auto pipeline = cuda::make_pipeline();


    //shared memory

    //TODO() make it dynamically sized
    __shared__ uint32_t sharedForGold[1024];
    __shared__ uint32_t sharedForSegm[1024];


    //for storing fp and fn sums to later accumulate it to global values
    __shared__ uint32_t fpSFnS[2];
    __shared__ uint32_t localBlockMetaData[20];

    __shared__ bool anyInGold[1];
    __shared__ bool anyInSegm[1];
    //__shared__ uint32_t reduction_s[32];
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    __shared__ int minMaxesInShmem[7];



    if ((threadIdx.x == 1) && (threadIdx.y == 1)) { fpSFnS[0] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 1)) { fpSFnS[1] = 0; };
    if ((threadIdx.x == 3) && (threadIdx.y == 1)) { anyInGold[1] = false; };
    if ((threadIdx.x == 4) && (threadIdx.y == 1)) {
        anyInSegm[1] = false;


    };



    sync(cta);

    /////////////////////////


    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        int xMeta = int(linIdexMeta % (metaData.metaXLength));
        int zMeta = int(floor((float)(linIdexMeta / (metaData.metaXLength * metaData.MetaYLength))));
        int yMeta = int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)));
        //reset
        isNotEmpty = false;
        sumFp = 0;
        sumFn = 0;
        anyInGold[0] = false;
        anyInSegm[0] = false;
        //iterating over data block
        sync(cta);
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.Ny && x < fbArgs.Nx) {

                    // resetting
                    sharedForGold[xLoc + yLoc * fbArgs.dbXLength] = 0;
                    sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] = 0;


                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = (zMeta + metaData.minZ) * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.Nz) {
                            //char* tensorslice;

                            //first array gold
                            bool goldBool = goldArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;
                            bool segmBool = segmArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;
                            //goldBool = true;

                            // setting bits
                            sharedForGold[xLoc + yLoc * fbArgs.dbXLength] |= goldBool << zLoc;
                            sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] |= segmBool << zLoc;
                            // setting value of local boolean marking that any of the entries was evaluated to true in either of arrays
                            isNotEmpty = (isNotEmpty || (goldBool || segmBool));
                            sumFp += (!goldBool && segmBool);
                            sumFn += (goldBool && !segmBool);
                            if (goldBool)  anyInGold[0] = true;
                            if (segmBool)  anyInSegm[0] = true;

                            //if (goldBool) {
                            //    printf("in kernel  gold x %d y %d z %d    xMeta %d yMeta %d zMeta %d counted ymeta %d linmeta %d \n", x, y, z,  xMeta, yMeta,zMeta
                            //        , int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)))
                            //    , linIdexMeta);
                            //}

                            //if (segmBool) {
                            //    printf("in kernel  segm  x %d y %d z %d    xMeta %d yMeta %d zMeta %d counted ymeta %d linmeta %d \n", x, y, z,  xMeta, yMeta, zMeta
                            //        , int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)))
                            //        , linIdexMeta);
                            //}


                        }
                    }
                }

                //if (sharedForGold[xLoc + yLoc * fbArgs.dbXLength] > 0) {
                //    printf("in kernel Metax %d yMeta %d zMeta %d linearLocal %d linIdexMeta %d column %d \n"
                //        , xMeta, yMeta, zMeta,  xLoc + yLoc * fbArgs.dbXLength, linIdexMeta
                //    , sharedForGold[xLoc + yLoc * fbArgs.dbXLength]);
                //}


            }
        }
        //reset local metadata
        if ((threadIdx.x < 20) && (threadIdx.y == 0)) {
            localBlockMetaData[threadIdx.x] = 0;
        }



        isNotEmpty = __syncthreads_or(isNotEmpty);
        //exporting to global memory
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.Ny && x < fbArgs.Nx) {

                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc] = sharedForGold[yLoc * 32 + xLoc];
                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc + metaData.mainArrXLength] = sharedForSegm[yLoc * 32 + xLoc];


                }
            }
        }

        //   sync(cta);



           //cuda::memcpy_async(cta, (&origArrs[linIdexMeta * metaData.mainArrSectionLength]) , (sharedForGold), sizeof(uint32_t) * cta.size(), barrier);
           //barrier.arrive_and_wait(); // Waits for all copies to complete


          // cuda::memcpy_async(cta, (&origArrs[linIdexMeta * metaData.mainArrSectionLength]), (sharedForGoldB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          //barrier.arrive_and_wait(); // Waits for all copies to complete

          //cuda::memcpy_async(cta, (&origArrs[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength]), (sharedForSegmB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          //barrier.arrive_and_wait(); // Waits for all copies to complete

          //cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength ]), (sharedForGoldB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          // barrier.arrive_and_wait(); // Waits for all copies to complete

          // cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength*1]), (sharedForSegmB), (sizeof(uint32_t) * blockDim.x * blockDim.y) , barrier);
          // barrier.arrive_and_wait(); // Waits for all copies to complete

          // cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength*2]), (sharedForGoldB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          // barrier.arrive_and_wait(); // Waits for all copies to complete

          // cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength*3]), (sharedForSegmB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
          // barrier.arrive_and_wait(); // Waits for all copies to complete

        sync(cta);



        /////adding the block and total number of the Fp's and Fn's
        sumFp = reduce(tile, sumFp, plus<uint32_t>());
        sumFn = reduce(tile, sumFn, plus<uint32_t>());
        //reusing shared memory and adding accumulated values from tiles
        if (tile.thread_rank() == 0) {
            sharedForGold[tile.meta_group_rank()] = sumFp;
            sharedForSegm[tile.meta_group_rank()] = sumFn;
        }
        sync(cta);//waiting so shared memory will be loaded evrywhere
        //on single thread we do last sum reduction
        auto active = coalesced_threads();

        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    printf("xMeta %d yMeta %d zMeta %d \n", xMeta, yMeta, zMeta);
        //}

        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty) {
            sharedForGold[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForGold[33] += sharedForGold[i];
                /*               if (sharedForGold[i]>0) {
                                   printf("adding sharedForGold[i] %d in gold \n ", sharedForGold[i]);
                               }*/

            };
            fpSFnS[0] += sharedForGold[33];// will be needed later for global set
            //metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 1] = sharedForGold[33];
            localBlockMetaData[1] = sharedForGold[33];

        }
        // if (isToBeExecutedOnActive(active, 1) && isNotEmpty) {
        if ((threadIdx.x == 0) && (threadIdx.y == 1) && isNotEmpty) {


            sharedForSegm[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForSegm[33] += sharedForSegm[i];
            };
            fpSFnS[1] += sharedForSegm[33];// will be needed later for global set
            //setting metadata
            localBlockMetaData[2] = sharedForSegm[33];

        }

        //marking as active
//FP pass
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty && anyInGold[0]) {
            localBlockMetaData[7] = 1;

        };
        //FN pass
        if ((threadIdx.x == 1) && (threadIdx.y == 0) && isNotEmpty && anyInSegm[0]) {
            localBlockMetaData[9] = 1;

        };


        //after we streamed over all block we save also information about indicies of the surrounding blocks - given they are in range if not UINT32_MAX will be saved
        //top



        setNeighbourBlocks(fbArgs, 3, 13, (zMeta > 0), (-(metaData.metaXLength * metaData.MetaYLength)), linIdexMeta, metaData, localBlockMetaData);//top
        setNeighbourBlocks(fbArgs, 4, 14, (zMeta < (metaData.MetaZLength - 1)), (metaData.metaXLength * metaData.MetaYLength), linIdexMeta, metaData, localBlockMetaData);//bottom

        setNeighbourBlocks(fbArgs, 6, 15, (xMeta > 0), (-1), linIdexMeta, metaData, localBlockMetaData);//left
        setNeighbourBlocks(fbArgs, 7, 16, (xMeta < (metaData.metaXLength - 1)), 1, linIdexMeta, metaData, localBlockMetaData);//right

        setNeighbourBlocks(fbArgs, 8, 17, (yMeta < (metaData.MetaYLength - 1)), metaData.metaXLength, linIdexMeta, metaData, localBlockMetaData);//anterior
        setNeighbourBlocks(fbArgs, 9, 18, (yMeta > 0), (-metaData.metaXLength), linIdexMeta, metaData, localBlockMetaData);//posterior

        if ((threadIdx.x < 20) && (threadIdx.y == 0)) {
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + threadIdx.x] = localBlockMetaData[threadIdx.x];
        };

        sync(cta); // just to reduce the warp divergence

        // copy metadata to global memory

        //cuda::memcpy_async(cta, &metaDataArr[linIdexMeta * metaData.metaDataSectionLength], (&localBlockMetaData[0]), (sizeof(uint32_t) * 20), barrier);
       // barrier.arrive_and_wait(); // Waits for all copies to complete

    }



    sync(cta);


    //setting global fp and fn
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[7]), fpSFnS[0]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[8]), fpSFnS[1]);

    };
}



/*
    a) we define offsets in the result list to have the results organized and avoid overwiting
    b) if metadata block is active we add it in the work queue
*/


/*
we add here to appropriate queue data  about metadata of blocks of intrest
minMaxesPos- marks in minmaxes the postion of global offset counter -12) global FP offset 13) global FnOffset
offsetMetadataArr- arrays from metadata holding data about result list offsets it can be either fbArgs.metaData.fpOffset or fbArgs.metaData.fnOffset
*/


#pragma once
__device__ inline void addToQueue(uint32_t linIdexMeta, uint8_t isGold
    , unsigned int fpFnLocCounter[1], uint32_t localWorkQueue[1600], uint32_t localOffsetQueue[1600], unsigned int localWorkQueueCounter[1]
    , uint8_t countIndexNumb, uint8_t isActiveIndexNumb, uint8_t offsetIndexNumb
    , uint32_t* metaDataArr, MetaDataGPU metaData, unsigned int* minMaxes, uint32_t* workQueue) {

    unsigned int count = metaDataArr[linIdexMeta * metaData.metaDataSectionLength + countIndexNumb];
    //given fp is non zero we need to  add this to local queue
    if (metaDataArr[linIdexMeta * metaData.metaDataSectionLength + isActiveIndexNumb] == 1) {

        // printf("adding to local in first meta pass linIdexMeta %d isGold %d isActiveIndexNumb %d \n  ", linIdexMeta, isGold, isActiveIndexNumb);

        count = atomicAdd_block(&fpFnLocCounter[0], count);
        unsigned int  old = atomicAdd_block(&localWorkQueueCounter[0], 1);
        //we check weather we still have space in shared memory
        if (old < 1590) {// so we still have space in shared memory
            // will be equal or above isGoldOffset  if it is gold pass
            localWorkQueue[old] = linIdexMeta + (isGoldOffset * isGold);
            localOffsetQueue[old] = uint32_t(count);
        }
        else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
            old = atomicAdd(&(minMaxes[9]), old);
            //workQueue
            workQueue[old] = linIdexMeta + (isGoldOffset * isGold);
            //and offset
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + offsetIndexNumb] = atomicAdd(&(minMaxes[12]), count);
        };
    }
}


#pragma once
template <typename PYO>
__global__ void firstMetaPrepareKernel(ForBoolKernelArgs<PYO> fbArgs
    , MetaDataGPU metaData, unsigned int* minMaxes, uint32_t* workQueue
    , uint32_t* origArrs, uint32_t* metaDataArr) {

    //////initializations
    thread_block cta = this_thread_block();
    char* tensorslice;// needed for iterations over 3d arrays
   //local offset counters  for fp and fn's
    __shared__ unsigned int fpFnLocCounter[1];
    // used to store the start position in global memory for whole block
    __shared__ unsigned int globalOffsetForBlock[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    //used as local work queue counter
    __shared__ unsigned int localWorkQueueCounter[1];
    //according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes so it will give us 1600 length of shared memory
    //so here we will store locally the calculated offsets and coordinates of meta data block of intrest marking also wheather we are  talking about gold or segmentation pass (fp or fn )
    __shared__ uint32_t localWorkQueue[1600];
    __shared__ uint32_t localOffsetQueue[1600];
    if ((threadIdx.x == 0)) {
        fpFnLocCounter[0] = 0;
    }
    if ((threadIdx.x == 1)) {
        localWorkQueueCounter[0] = 0;
    }
    if ((threadIdx.x == 2)) {
        globalWorkQueueCounter[0] = 0;
    }
    if ((threadIdx.x == 3)) {
        globalOffsetForBlock[0] = 0;
    }
    sync(cta);


    // classical grid stride loop - in case of unlikely event we will run out of space we will empty it prematurly
    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x * blockDim.x + threadIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += blockDim.x * gridDim.x) {

        // if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
           //  printf("in first meta pass linIdexMeta %d blockIdx.x %d blockDim.x %d metaData.totalMetaLength %d threadIdx.x %d \n  ", linIdexMeta, blockIdx.x, blockDim.x, metaData.totalMetaLength, threadIdx.x );
         //}

         //goldpass
        addToQueue(linIdexMeta, 0
            , fpFnLocCounter, localWorkQueue, localOffsetQueue, localWorkQueueCounter
            , 1, 9, 6
            , metaDataArr, metaData, minMaxes, workQueue);
        //segmPass
        addToQueue(linIdexMeta, 1
            , fpFnLocCounter, localWorkQueue, localOffsetQueue, localWorkQueueCounter
            , 2, 7, 5
            , metaDataArr, metaData, minMaxes, workQueue);



        /*       addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fpOffset, fbArgs.metaData.fpCount, 0, fbArgs.metaData.isActiveSegm, fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);
               addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fnOffset, fbArgs.metaData.fnCount, 1, fbArgs.metaData.isActiveGold, fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);*/
    }
    sync(cta);
    if ((threadIdx.x == 0)) {
        globalOffsetForBlock[0] = atomicAdd(&(minMaxes[12]), (fpFnLocCounter[0]));

        /* if (fpFnLocCounter[0]>0) {
             printf("\n in meta first pass global offset %d  locCounter %d \n  ", globalOffsetForBlock[0], fpFnLocCounter[0]);
         }*/
    };
    if ((threadIdx.x == 1)) {
        if (localWorkQueueCounter[0] > 0) {
            globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
        }
    }
    sync(cta);

    //exporting to global work queue
    //cooperative_groups::memcpy_async(cta, (&workQueue[globalWorkQueueCounter[0]]), (localWorkQueue), (sizeof(uint32_t) * localWorkQueueCounter[0]));


    //setting offsets
    for (uint32_t i = threadIdx.x; i < localWorkQueueCounter[0]; i += blockDim.x) {
        workQueue[globalWorkQueueCounter[0] + i] = localWorkQueue[i];

        /*        printf("FFIrst meta pass lin meta to Work Q %d is gold %d to spot %d  \n "
            , localWorkQueue[i] - isGoldOffset*(localWorkQueue[i] >= isGoldOffset)
                , (localWorkQueue[i] >= isGoldOffset), globalWorkQueueCounter[0] + i);*/

                //FP pass
        if (localWorkQueue[i] >= isGoldOffset) {
            metaDataArr[(localWorkQueue[i] - isGoldOffset) * metaData.metaDataSectionLength + 5] = localOffsetQueue[i] + globalOffsetForBlock[0];
            //printf("fp offset lin meta %d total offset  %d  global part %d local part %d \n "
            //    , localWorkQueue[i] - isGoldOffset
            //    , localOffsetQueue[i] + globalOffsetForBlock[0]
            //, globalOffsetForBlock[0]
            //, localOffsetQueue[i]);

        }
        //FN pass
        else {
            metaDataArr[(localWorkQueue[i]) * metaData.metaDataSectionLength + 6] = localOffsetQueue[i] + globalOffsetForBlock[0];
            //printf("fn offset lin meta %d total offset  %d  global part %d local part %d \n "
            //    , localWorkQueue[i]
            //    , localOffsetQueue[i] + globalOffsetForBlock[0]
            //    , globalOffsetForBlock[0]
            //    , localOffsetQueue[i]);

        };

        //sync(cta);


    }



};







// helper functions and utilities to work with CUDA from https://github.com/NVIDIA/cuda-samples








/*
*/
#pragma once
template <typename TCC>
inline ForBoolKernelArgs<TCC> getArgsForKernel(ForFullBoolPrepArgs<TCC>& mainFunArgs
    , int& warpsNumbForMainPass, int& blockForMainPass
    , const int xLen, const int yLen, const int zLen, cudaStream_t stream
) {


    mainFunArgs.Nx = xLen;
    mainFunArgs.Ny = yLen;
    mainFunArgs.Nz = zLen;

    //main arrays allocations
    //TCC* goldArrPointer;
    //TCC* segmArrPointer;
    //size_t sizeMainArr = (sizeof(T) * WIDTH * HEIGHT * DEPTH);
    //size_t sizeMainArr = (sizeof(TCC) * xLen * yLen * zLen);
    //array3dWithDimsGPU<TCC> goldArr = allocateMainArray(goldArrPointer, mainFunArgs.goldArr.arrP, xLen, yLen, zLen, stream);
    //array3dWithDimsGPU<TCC> segmArr = allocateMainArray(segmArrPointer, mainFunArgs.segmArr.arrP, xLen, yLen, zLen, stream);
    unsigned int* minMaxes;
    size_t sizeminMaxes = sizeof(unsigned int) * 20;
    cudaMallocAsync(&minMaxes, sizeminMaxes, stream);
    ForBoolKernelArgs<TCC> res;

    res.Nx = xLen;
    res.Ny = yLen;
    res.Nz = zLen;

    res.metaData = allocateMetaDataOnGPU(mainFunArgs.metaData, minMaxes);
    res.metaData.minMaxes = minMaxes;
    res.minMaxes = minMaxes;
    res.numberToLookFor = mainFunArgs.numberToLookFor;
    res.dbXLength = 32;
    res.dbYLength = warpsNumbForMainPass;
    res.dbZLength = 32;

    //printf("in setting bool args ylen %d dbYlen %d calculated meta %d  \n ", yLen, res.dbYLength, int(ceil(yLen / res.dbYLength)));
    res.metaData.metaXLength = int(ceil(xLen / res.dbXLength));
    res.metaData.MetaYLength = int(ceil(yLen / res.dbYLength));;
    res.metaData.MetaZLength = int(ceil(zLen / res.dbZLength));;
    res.metaData.minX = 0;
    res.metaData.minY = 0;
    res.metaData.minZ = 0;
    res.metaData.maxX = res.metaData.metaXLength;
    res.metaData.maxY = res.metaData.MetaYLength;
    res.metaData.maxZ = res.metaData.MetaZLength;

    res.metaData.totalMetaLength = res.metaData.metaXLength * res.metaData.MetaYLength * res.metaData.MetaZLength;
    res.goldArr = mainFunArgs.goldArr;
    res.segmArr = mainFunArgs.segmArr;


    return res;
}







#pragma once
template <typename ZZR>
inline MetaDataGPU allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs, cudaStream_t stream) {
    ////reduced arrays
    uint32_t* origArr;
    uint32_t* metaDataArr;
    uint32_t* workQueue;
    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.minMaxes, size, cudaMemcpyDeviceToHost, stream);

    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int xRange = cpuArgs.metaData.minMaxes[1] - cpuArgs.metaData.minMaxes[2] + 1;
    unsigned int yRange = cpuArgs.metaData.minMaxes[3] - cpuArgs.metaData.minMaxes[4] + 1;
    unsigned int zRange = cpuArgs.metaData.minMaxes[5] - cpuArgs.metaData.minMaxes[6] + 1;
    unsigned int totalMetaLength = (xRange) * (yRange) * (zRange);
    //updating size informations
    gpuArgs.metaData.metaXLength = xRange;
    gpuArgs.metaData.MetaYLength = yRange;
    gpuArgs.metaData.MetaZLength = zRange;
    gpuArgs.metaData.totalMetaLength = totalMetaLength;
    //saving min maxes
    gpuArgs.metaData.maxX = cpuArgs.metaData.minMaxes[1];
    gpuArgs.metaData.minX = cpuArgs.metaData.minMaxes[2];
    gpuArgs.metaData.maxY = cpuArgs.metaData.minMaxes[3];
    gpuArgs.metaData.minY = cpuArgs.metaData.minMaxes[4];
    gpuArgs.metaData.maxZ = cpuArgs.metaData.minMaxes[5];
    gpuArgs.metaData.minZ = cpuArgs.metaData.minMaxes[6];

    //allocating needed memory
    // main array
    unsigned int mainArrXLength = gpuArgs.dbXLength * gpuArgs.dbYLength;
    unsigned int mainArrSectionLength = (mainArrXLength * 2);
    gpuArgs.metaData.mainArrXLength = mainArrXLength;
    gpuArgs.metaData.mainArrSectionLength = mainArrSectionLength;

    size_t sizeB = totalMetaLength * mainArrSectionLength * sizeof(uint32_t);
    //cudaMallocAsync(&mainArr, sizeB, 0);
    size_t sizeorigArr = totalMetaLength * (mainArrXLength * 2) * sizeof(uint32_t);
    cudaMallocAsync(&origArr, sizeorigArr, stream);
    size_t sizemetaDataArr = totalMetaLength * (20) * sizeof(uint32_t) + 100;
    cudaMallocAsync(&metaDataArr, sizemetaDataArr, stream);
    size_t sizeC = (totalMetaLength * 2 * sizeof(uint32_t) + 50);
    cudaMallocAsync(&workQueue, sizeC, stream);
    gpuArgs.origArrsPointer = origArr;
    gpuArgs.metaDataArrPointer = metaDataArr;
    gpuArgs.workQueuePointer = workQueue;
    return gpuArgs.metaData;
};




/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline int allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs, cudaStream_t stream, bool resIterNeeded) {

    int32_t* resultListPointerMeta;
    int32_t* resultListPointerLocal;
    int32_t* resultListPointerIterNumb;
    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    //free no longer needed arrays
    //cudaFreeAsync(gpuArgs.goldArr.arrP, stream);
    //cudaFreeAsync(gpuArgs.segmArr.arrP, stream);

    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost, stream);

    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];
    size = sizeof(int32_t) * (fpPlusFn + 50);


    cudaMallocAsync(&resultListPointerLocal, size, stream);
    cudaMallocAsync(&resultListPointerIterNumb, size, stream);
    cudaMallocAsync(&resultListPointerMeta, size, stream);


// we will allocate only if we need
    if (resIterNeeded) {
        auto options =
            torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA, 0)
            .requires_grad(false);

        auto resultListPointerIterNumbTensor = torch::empty((fpPlusFn + 50), options);
        gpuArgs.resultListPointerIterNumbTensor = resultListPointerIterNumbTensor;
    }


    auto xRange = gpuArgs.metaData.metaXLength;
    auto yRange = gpuArgs.metaData.MetaYLength;
    auto zRange = gpuArgs.metaData.MetaZLength;


    size_t sizeB = gpuArgs.metaData.totalMetaLength * gpuArgs.metaData.mainArrSectionLength * sizeof(uint32_t);

    //printf("size of reduced main arr %d total meta len %d mainArrSectionLen %d  \n", sizeB, metaData.totalMetaLength, metaData.mainArrSectionLength);

    cudaMallocAsync(&mainArrAPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrAPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, stream);


    cudaMallocAsync(&mainArrBPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrBPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, stream);

    //just in order set it to 0
    int32_t* resultListPointerMetaCPU = (int32_t*)calloc(fpPlusFn + 50, sizeof(int32_t));
    cudaMemcpyAsync(resultListPointerMeta, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(resultListPointerIterNumb, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, stream);
    free(resultListPointerMetaCPU);

    gpuArgs.resultListPointerMeta = resultListPointerMeta;
    gpuArgs.resultListPointerLocal = resultListPointerLocal;
    gpuArgs.resultListPointerIterNumb = resultListPointerIterNumb;

    //fbArgs.origArrsPointer = origArrsPointer;
    gpuArgs.mainArrAPointer = mainArrAPointer;
    gpuArgs.mainArrBPointer = mainArrBPointer;


    return fpPlusFn;
};








#pragma once
template <typename T>
inline void  copyResultstoCPU(ForBoolKernelArgs<T>& gpuArgs, ForFullBoolPrepArgs<T>& cpuArgs, cudaStream_t stream) {


    ////copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost, stream);
    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];
    size = sizeof(uint32_t) * (fpPlusFn + 50);

    //uint32_t* resultListPointerMeta = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    //uint32_t* resultListPointerLocal = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    //uint32_t* resultListPointerIterNumb = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));

    cpuArgs.resultListPointerMeta = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));;
    cpuArgs.resultListPointerLocalCPU = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));;
    cpuArgs.resultListPointerIterNumb = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));;

    cudaMemcpyAsync(cpuArgs.resultListPointerMeta, gpuArgs.resultListPointerMeta, size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(cpuArgs.resultListPointerLocalCPU, gpuArgs.resultListPointerLocal, size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(cpuArgs.resultListPointerIterNumb, gpuArgs.resultListPointerIterNumb, size, cudaMemcpyDeviceToHost, stream);


};



/*
gettinng  array for dilatations
basically arrays will alternate between iterations once one will be source other target then they will switch - we will decide upon knowing
wheather the iteration number is odd or even
*/
#pragma once
template <typename TXPI>
inline __device__ uint32_t* getSourceReduced(const ForBoolKernelArgs<TXPI>& fbArgs, const int(&iterationNumb)[1]) {


    if ((iterationNumb[0] & 1) == 0) {
        return fbArgs.mainArrAPointer;

    }
    else {
        return fbArgs.mainArrBPointer;
    }


}


/*
gettinng target array for dilatations
*/
#pragma once
template <typename TXPPI>
inline __device__ uint32_t* getTargetReduced(const ForBoolKernelArgs<TXPPI>& fbArgs, const  int(&iterationNumb)[1]) {

    if ((iterationNumb[0] & 1) == 0) {
        //printf(" BB ");

        return fbArgs.mainArrBPointer;

    }
    else {
        // printf(" AA ");

        return fbArgs.mainArrAPointer;

    }

}


/*
dilatation up and down - using bitwise operators
*/
#pragma once
inline __device__ uint32_t bitDilatate(const uint32_t& x) {
    return ((x) >> 1) | (x) | ((x) << 1);
}

/*
return 1 if at given position of given number bit is set otherwise 0
*/
#pragma once
inline __device__ uint32_t isBitAt(const uint32_t& numb, const int pos) {
    return (numb & (1 << (pos)));
}

#pragma once
inline uint32_t isBitAtCPU(const uint32_t& numb, const int pos) {
    return (numb & (1 << (pos)));
}







/*
5)Main block
    a) we define the work queue iteration - so we divide complete work queue into parts  and each thread block analyzes its own part - one data block at a textLinesFromStrings
    b) we load values of data block into shared memory  and immidiately do the bit wise up and down dilatations, and mark booleans needed to establish is the datablock full
    c) synthreads - left,right, anterior,posterior dilatations...
    d) add the dilatated info into dilatation array and padding info from dilatation to global memory
    e) if block is to be validated we check is there is in the point of currently coverd voxel some voxel in other mas if so we add it to the result list and increment local reult counter
    f) syncgrid()
6)analyze padding
    we iterate over work queue as in 5
    a) we load into shared memory information from padding from blocks all around the block of intrest checking for boundary conditions
    b) we save data of dilatated voxels into dilatation array making sure to synchronize appropriately in the thread block
    c) we analyze the positive entries given the block is to be validated  so we check is such entry is already in dilatation mask if not is it in other mask if first no and second yes we add to the result
    d) also given any positive entry we set block as to be activated simple sum reduction should be sufficient
    e) sync grid
*/
template <typename TKKI>
inline __global__ void mainPassKernel(ForBoolKernelArgs<TKKI> fbArgs) {



    thread_block cta = cooperative_groups::this_thread_block();

    grid_group grid = cooperative_groups::this_grid();

    /*
    * according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes
    main shared memory spaces
    0-1023 : sourceShmem
    1024-2047 : resShmem
    2048-3071 : first register space
    3072-4095 : second register space
    4096-  4127: small 32 length resgister 3 space
    4128-4500 (372 length) : place for local work queue in dilatation kernels
    */
    // __shared__ uint32_t mainShmem[lengthOfMainShmem];
    __shared__ uint32_t mainShmem[lengthOfMainShmem];
    //cuda::associate_access_property(&mainShmem, cuda::access_property::shared{});



    constexpr size_t stages_count = 2; // Pipeline stages number

    // Allocate shared storage for a two-stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;

    //cuda::pipeline<cuda::thread_scope_thread>  pipeline = cuda::make_pipeline(cta, &shared_state);
    cuda::pipeline<cuda::thread_scope_block>  pipeline = cuda::make_pipeline(cta, &shared_state);



    //usefull for iterating through local work queue
    __shared__ bool isGoldForLocQueue[localWorkQueLength];
    // holding data about paddings


    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
    __shared__ bool isAnythingInPadding[6];

    __shared__ bool isBlockFull[2];

    __shared__ uint32_t lastI[1];


    //variables needed for all threads
    __shared__ int iterationNumb[1];
    __shared__ unsigned int globalWorkQueueOffset[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    __shared__ unsigned int localWorkQueueCounter[1];
    // keeping data wheather gold or segmentation pass should continue - on the basis of global counters

    __shared__ unsigned int localTotalLenthOfWorkQueue[1];
    //counters for per block number of results added in this iteration
    __shared__ unsigned int localFpConter[1];
    __shared__ unsigned int localFnConter[1];

    __shared__ unsigned int blockFpConter[1];
    __shared__ unsigned int blockFnConter[1];

    __shared__ unsigned int fpFnLocCounter[1];

    //result list offset - needed to know where to write a result in a result list
    __shared__ unsigned int resultfpOffset[1];
    __shared__ unsigned int resultfnOffset[1];

    __shared__ unsigned int worQueueStep[1];


    /* will be used to store all of the minMaxes varibles from global memory (from 7 to 11)
    0 : global FP count;
    1 : global FN count;
    2 : workQueueCounter
    3 : resultFP globalCounter
    4 : resultFn globalCounter
    */
    __shared__ unsigned int localMinMaxes[5];

    /* will be used to store all of block metadata
  nothing at  0 index
 1 :fpCount
 2 :fnCount
 3 :fpCounter
 4 :fnCounter
 5 :fpOffset
 6 :fnOffset
 7 :isActiveGold
 8 :isFullGold
 9 :isActiveSegm
 10 :isFullSegm
 11 :isToBeActivatedGold
 12 :isToBeActivatedSegm
 12 :isToBeActivatedSegm
//now linear indexes of the blocks in all sides - if there is no block in given direction it will equal UINT32_MAX
 13 : top
 14 : bottom
 15 : left
 16 : right
 17 : anterior
 18 : posterior
    */

    __shared__ uint32_t localBlockMetaData[40];

    /*
 //now linear indexes of the previous block in all sides - if there is no block in given direction it will equal UINT32_MAX
 0 : top
 1 : bottom
 2 : left
 3 : right
 4 : anterior
 5 : posterior
    */


    /////used mainly in meta passes

//    __shared__ unsigned int fpFnLocCounter[1];
    __shared__ bool isGoldPassToContinue[1];
    __shared__ bool isSegmPassToContinue[1];





    //initializations and loading
    if (threadIdx.x == 9 && threadIdx.y == 0) { iterationNumb[0] = -1; };
    if (threadIdx.x == 11 && threadIdx.y == 0) {
        isGoldPassToContinue[0] = true;
    };
    if (threadIdx.x == 12 && threadIdx.y == 0) {
        isSegmPassToContinue[0] = true;

    };


    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number
    sync(cta);

    do {

        for (uint8_t isPaddingPass = 0; isPaddingPass < 2; isPaddingPass++) {


            /////////////////////////****************************************************************************************************************
            /////////////////////////****************************************************************************************************************
            /////////////////////////****************************************************************************************************************
            /////////////////////////****************************************************************************************************************
            /////////////////////////****************************************************************************************************************
            /// dilataions

    //initial cleaning  and initializations include loading min maxes
            if (threadIdx.x == 7 && threadIdx.y == 0 && !isPaddingPass) {
                iterationNumb[0] += 1;
            };

            if (threadIdx.x == 6 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            };

            if (threadIdx.x == 1 && threadIdx.y == 0) {
                blockFpConter[0] = 0;
            };
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                blockFnConter[0] = 0;
            };
            if (threadIdx.x == 3 && threadIdx.y == 0) {
                localFpConter[0] = 0;
            };
            if (threadIdx.x == 4 && threadIdx.y == 0) {
                localFnConter[0] = 0;
            };
            if (threadIdx.x == 9 && threadIdx.y == 0) {
                isBlockFull[0] = true;
            };
            if (threadIdx.x == 9 && threadIdx.y == 1) {
                isBlockFull[1] = true;
            };

            if (threadIdx.x == 10 && threadIdx.y == 0) {
                fpFnLocCounter[0] = 0;
            };



            if (threadIdx.x == 0 && threadIdx.y == 0) {
                localTotalLenthOfWorkQueue[0] = fbArgs.minMaxes[9];
                globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
                worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
            };

            if (threadIdx.y == 1) {
                cooperative_groups::memcpy_async(cta, (&localMinMaxes[0]), (&fbArgs.minMaxes[7]), cuda::aligned_size_t<4>(sizeof(unsigned int) * 5));
            }

            sync(cta);

            /// load work QueueData into shared memory
            for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {

                //grid stride loop - sadly most of threads will be idle
               ///////// loading to work queue
                if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                    for (uint16_t ii = cta.thread_rank(); ii < worQueueStep[0]; ii += cta.size()) {

                        mainShmem[startOfLocalWorkQ + ii] = fbArgs.workQueuePointer[bigloop + ii];
                        isGoldForLocQueue[ii] = (mainShmem[startOfLocalWorkQ + ii] >= isGoldOffset);
                        mainShmem[startOfLocalWorkQ + ii] = mainShmem[startOfLocalWorkQ + ii] - isGoldOffset * isGoldForLocQueue[ii];


                    }

                }
                //now all of the threads in the block needs to have the same i value so we will increment by 1 we are preloading to the pipeline block metaData
                ////##### pipeline Step 0

                sync(cta);




                //loading metadata
                pipeline.producer_acquire();
                if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                    cuda::memcpy_async(cta, (&localBlockMetaData[0]),
                        (&fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ] * fbArgs.metaData.metaDataSectionLength])
                        , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

                }
                pipeline.producer_commit();


                sync(cta);

                for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {




                    if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {



                        pipeline.producer_acquire();
                        cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
                            mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                            cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength), pipeline);
                        pipeline.producer_commit();

                        //just so pipeline will work well
                        pipeline.consumer_wait();



                        pipeline.consumer_release();
                        sync(cta);

                        ///////// step 1 load top and process main data
                                        //load top
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 13]
                                * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process main
                        pipeline.consumer_wait();
                        //marking weather block is already full and no more dilatations are possible
                        if (__popc(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) < 32) {
                            isBlockFull[i & 1] = false;
                        }
                        mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);
                        pipeline.consumer_release();

                        ///////// step 2 load bottom and process top
                                        //load bottom
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
                            cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 14]
                                * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process top
                        pipeline.consumer_wait();


                        if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
                            if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], 0)) {
                                // printf("setting padding top val %d \n ", isAnythingInPadding[0]);
                                isAnythingInPadding[0] = true;
                            };
                            // if in bit of intrest of neighbour block is set
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32] >> 31) & 1) << 0;
                        }

                        pipeline.consumer_release();
                        sync(cta);

                        /////////// step 3 load right  process bottom
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 16] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process bototm
                        pipeline.consumer_wait();


                        if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
                            if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], 31)) {
                                isAnythingInPadding[1] = true;
                            };
                            // if in bit of intrest of neighbour block is set
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] >> 0) & 1) << 31;
                        }



                        /*  dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
                              , 0, 31
                              , begSecRegShmem, i);*/

                        pipeline.consumer_release();
                        /////////// step 4 load left process right
                                        //load left
                        pipeline.producer_acquire();
                        if (mainShmem[startOfLocalWorkQ + i] > 0) {
                            cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[(mainShmem[startOfLocalWorkQ + i] - 1) * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process right
                        pipeline.consumer_wait();

                        if (threadIdx.x == (fbArgs.dbXLength - 1)) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[3] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begfirstRegShmem + (threadIdx.y * 32)];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x + 1) + (threadIdx.y) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }

                        pipeline.consumer_release();
                        sync(cta);
                        /////// step 5 load anterior process left
                                        //load anterior
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {

                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 17] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process left
                        pipeline.consumer_wait();

                        // so we first check for corner cases
                        if (threadIdx.x == 0) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 15] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[2] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begSecRegShmem + 31 + threadIdx.y * 32];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x - 1) + (threadIdx.y) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }


                        pipeline.consumer_release();
                        sync(cta);

                        /////// step 6 load posterior process anterior
                                        //load posterior
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {


                            cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 18] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();

                        //process anterior
                        pipeline.consumer_wait();

                        // so we first check for corner cases
                        if (threadIdx.y == (fbArgs.dbYLength - 1)) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[4] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begfirstRegShmem + threadIdx.x];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x) + (threadIdx.y + 1) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }


                        pipeline.consumer_release();
                        sync(cta);

                        /////// step 7
                                       //load reference if needed or data for next iteration if there is such
                                        //process posterior, save data from res shmem to global memory also we mark weather block is full
                        pipeline.producer_acquire();

                        //if block should be validated we load data for validation
                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                        > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &fbArgs.origArrsPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (isGoldForLocQueue[i])], //we look for
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);

                        }
                        else {//if we are not validating we immidiately start loading data for next loop
                            if (i + 1 < worQueueStep[0]) {
                                cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                                    (&fbArgs.metaDataArrPointer[(mainShmem[startOfLocalWorkQ + 1 + i])
                                        * fbArgs.metaData.metaDataSectionLength])
                                    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


                            }
                        }


                        pipeline.producer_commit();

                        //processPosteriorAndSaveResShmem

                        pipeline.consumer_wait();
                        //dilatate posterior


                        // so we first check for corner cases
                        if (threadIdx.y == 0) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[5] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begSecRegShmem + threadIdx.x + (fbArgs.dbYLength - 1) * 32];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x) + (threadIdx.y - 1) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }

                        //now all data should be properly dilatated we save it to global memory
                        //try save target reduced via mempcy async ...


                        //cuda::memcpy_async(cta,
                        //    &getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])]
                        //    , (&mainShmem[begResShmem]),
                        //    cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                        //    , pipeline);



                        getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
                            + threadIdx.x + threadIdx.y * 32]
                            = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];





                        pipeline.consumer_release();

                        sync(cta);

                        //////// step 8 basically in order to complete here anyting the count need to be bigger than counter
                                                      // loading for next block if block is not to be validated it was already done earlier
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                            > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                            if (i + 1 < worQueueStep[0]) {


                                cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                                    (&fbArgs.metaDataArrPointer[(mainShmem[startOfLocalWorkQ + 1 + i])
                                        * fbArgs.metaData.metaDataSectionLength])
                                    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

                            }
                        }
                        pipeline.producer_commit();




                        sync(cta);

                        //validation - so looking for newly covered voxel for opposite array so new fps or new fns
                        pipeline.consumer_wait();

                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                            > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                                        // now we look through bits and when some is set we call it a result
#pragma unroll
                            for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
                                //if any bit here is set it means it should be added to result list
                                if (isBitAt(mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    && !isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    && isBitAt(mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    ) {

                                    //just re
                                    mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = 0;
                                    ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
                                    if (isGoldForLocQueue[i]) {
                                        mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFpConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 6] + localBlockMetaData[(i & 1) * 20 + 3]);

                                    }
                                    else {

                                        mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 5] + localBlockMetaData[(i & 1) * 20 + 4]);


                                    };
                                    //   add results to global memory
                                    //we add one gere jjust to distinguish it from empty result
                                    fbArgs.resultListPointerMeta[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(mainShmem[startOfLocalWorkQ + i] + (isGoldOffset * isGoldForLocQueue[i]) + 1);
                                    fbArgs.resultListPointerLocal[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x));
                                    fbArgs.resultListPointerIterNumb[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(iterationNumb[0]);




                                }

                            };

                        }
                        /////////
                        pipeline.consumer_release();

                        /// /// cleaning

                        sync(cta);

                        if (threadIdx.x == 9 && threadIdx.y == 2) {// this is how it is encoded wheather it is gold or segm block

         //executed in case of previous block
                            if (isBlockFull[i & 1] && i >= 0) {
                                //setting data in metadata that block is full
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 10 - (isGoldForLocQueue[i] * 2)] = true;
                            }
                            //resetting for some reason  block 0 gets as full even if it should not ...
                            isBlockFull[i & 1] = true;// mainShmem[startOfLocalWorkQ + i]>0;//!isPaddingPass;
                        };




                        //we do it only for non padding pass
                        if (threadIdx.x < 6 && threadIdx.y == 1 && !isPaddingPass) {
                            //executed in case of previous block
                            if (i >= 0) {

                                if (localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x] < isGoldOffset) {

                                    if (isAnythingInPadding[threadIdx.x]) {
                                        // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
                                        fbArgs.metaDataArrPointer[localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x] * fbArgs.metaData.metaDataSectionLength + 12 - isGoldForLocQueue[i]] = 1;
                                    }

                                }
                            }
                            isAnythingInPadding[threadIdx.x] = false;
                        };






                        if (threadIdx.x == 7 && threadIdx.y == 0) {
                            //this will be executed only if fp or fn counters are bigger than 0 so not during first pass
                            if (localFpConter[0] > 0) {
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 3] += localFpConter[0];

                                blockFpConter[0] += localFpConter[0];
                                localFpConter[0] = 0;
                            }


                        };
                        if (threadIdx.x == 8 && threadIdx.y == 0) {

                            if (localFnConter[0] > 0) {
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 4] += localFnConter[0];

                                blockFnConter[0] += localFnConter[0];
                                localFnConter[0] = 0;
                            }
                        };

                        sync(cta);

                    }
                }

                //here we are after all of the blocks planned to be processed by this block are

                // just for pipeline to work
                pipeline.consumer_wait();



                pipeline.consumer_release();

            }

            sync(cta);

            //     updating global counters
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                if (blockFpConter[0] > 0) {
                    atomicAdd(&(fbArgs.minMaxes[10]), (blockFpConter[0]));
                }
            };
            if (threadIdx.x == 1 && threadIdx.y == 0) {
                if (blockFnConter[0] > 0) {
                    //if (blockFnConter[0]>10) {
                    //    printf("Fn %d  ", blockFnConter[0]);
                    //}
                    atomicAdd(&(fbArgs.minMaxes[11]), (blockFnConter[0]));
                }
            };

            // in first thread block we zero work queue counter
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                if (blockIdx.x == 0) {

                    fbArgs.minMaxes[9] = 0;
                }
            };

            grid.sync();
            /////////////////////////****************************************************************************************************************
/////////////////////////****************************************************************************************************************
/////////////////////////****************************************************************************************************************
/////////////////////////****************************************************************************************************************
/////////////////////////****************************************************************************************************************
/// metadata pass










            // preparation loads
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                fpFnLocCounter[0] = 0;
            }
            if (threadIdx.x == 1 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            }
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            }
            if (threadIdx.x == 3 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;

            }

            if (threadIdx.x == 0 && threadIdx.y == 1) {

                isGoldPassToContinue[0]
                    = ((fbArgs.minMaxes[7] * fbArgs.robustnessPercent) > fbArgs.minMaxes[10]);

            };

            if (threadIdx.x == 0 && threadIdx.y == 1) {

                isSegmPassToContinue[0]
                    = ((fbArgs.minMaxes[8] * fbArgs.robustnessPercent) > fbArgs.minMaxes[11]);
            };


            __syncthreads();

            /////////////////////////////////

            for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
                ; linIdexMeta <= fbArgs.metaData.totalMetaLength
                ; linIdexMeta += (blockDim.x * blockDim.y * gridDim.x)
                ) {


                if (isPaddingPass == 0) {

                    //goldpass
                    if (isGoldPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 11]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 8]) {

                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
                        //setting to be activated to 0
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 11] = 0;
                        //setting active to 1
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7] = 1;


                    };

                }
                //contrary to number it is when we are not in padding pass
                else {
                    //gold pass
                    if (isGoldPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 8]) {

                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);

                    };

                }
            }

            __syncthreads();

            if (localWorkQueueCounter[0] > 0) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    globalWorkQueueCounter[0] = atomicAdd(&(fbArgs.minMaxes[9]), (localWorkQueueCounter[0]));


                }
                __syncthreads();
                for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
                    fbArgs.workQueuePointer[globalWorkQueueCounter[0] + linI] = mainShmem[linI];
                }
                __syncthreads();

            }

            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {

                localWorkQueueCounter[0] = 0;
            }
            __syncthreads();

            for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
                ; linIdexMeta <= fbArgs.metaData.totalMetaLength
                ; linIdexMeta += (blockDim.x * blockDim.y * gridDim.x)
                ) {


                if (isPaddingPass == 0) {

                    //segm pass
                    if ((isSegmPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 12]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 10])) {



                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;

                        //setting to be activated to 0
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 12] = 0;
                        //setting active to 1
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9] = 1;

                    }

                }
                //contrary to number it is when we are not in padding pass
                else {
                    //segm pass
                    if ((isSegmPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 10])) {



                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
                    }

                }
            }
            __syncthreads();

            if (localWorkQueueCounter[0] > 0) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    globalWorkQueueCounter[0] = atomicAdd(&(fbArgs.minMaxes[9]), (localWorkQueueCounter[0]));


                }
                __syncthreads();
                for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
                    fbArgs.workQueuePointer[globalWorkQueueCounter[0] + linI] = mainShmem[linI];

                }

            }



            grid.sync();
        }



    } while (isGoldPassToContinue[0] || isSegmPassToContinue[0]);


    //setting global iteration number to local one
    if (blockIdx.x == 0) {
        if (threadIdx.x == 2 && threadIdx.y == 0) {
            fbArgs.metaData.minMaxes[13] = iterationNumb[0];
        }
    }
}








/*
get data from occupancy calculator API used to get optimal number of thread blocks and threads per thread block
*/
template <typename T>
inline occupancyCalcData getOccupancy() {

    occupancyCalcData res;

    int blockSize; // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize; // The actual grid size needed, based on input size

    // for min maxes kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)getMinMaxes<T>,
        0);
    res.warpsNumbForMinMax = blockSize / 32;
    res.blockSizeForMinMax = minGridSize;

    // for min maxes kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    res.warpsNumbForboolPrepareKernel = blockSize / 32;
    res.blockSizeFoboolPrepareKernel = minGridSize;
    // for first meta pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    res.theadsForFirstMetaPass = blockSize;
    res.blockForFirstMetaPass = minGridSize;
    //for main pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)mainPassKernel<T>,
        0);
    res.warpsNumbForMainPass = blockSize / 32;
    res.blockForMainPass = minGridSize;

    // res.blockForMainPass = 1;
     //res.blockForMainPass = 136;
     //res.warpsNumbForMainPass = 8;

    //printf("warpsNumbForMainPass %d blockForMainPass %d  ", res.warpsNumbForMainPass, res.blockForMainPass);
    return res;
}



/*
TODO consider representing as a CUDA graph
executing Algorithm as CUDA graph  based on official documentation and
https://codingbyexample.com/2020/09/25/cuda-graph-usage/
*/
#pragma once
template <typename T>
ForBoolKernelArgs<T> executeHausdoff(ForFullBoolPrepArgs<T>& fFArgs, const int WIDTH, const int HEIGHT, const int DEPTH, occupancyCalcData& occData,
    cudaStream_t stream, bool resToSave, float robustnessPercent, bool resIterNeeded) {
    fFArgs.robustnessPercent = robustnessPercent;
    T* goldArrPointer = (T*)fFArgs.goldArr.data_ptr();
    T* segmArrPointer = (T*)fFArgs.segmArr.data_ptr();

        ForBoolKernelArgs<T> fbArgs = getArgsForKernel<T>(fFArgs, occData.warpsNumbForMainPass, occData.blockForMainPass, WIDTH, HEIGHT, DEPTH, stream);

    getMinMaxes << <occData.blockSizeForMinMax, dim3(32, occData.warpsNumbForMinMax) >> > (fbArgs, fbArgs.minMaxes
        , goldArrPointer
        , segmArrPointer
        , fbArgs.metaData);



    fbArgs.metaData = allocateMemoryAfterMinMaxesKernel(fbArgs, fFArgs, stream);
    fbArgs.robustnessPercent = robustnessPercent;
    boolPrepareKernel << <occData.blockSizeFoboolPrepareKernel, dim3(32, occData.warpsNumbForboolPrepareKernel) >> > (
        fbArgs, fbArgs.metaData, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer
        , goldArrPointer
        , segmArrPointer
        , fbArgs.minMaxes);


    int fpPlusFn = allocateMemoryAfterBoolKernel(fbArgs, fFArgs, stream, resIterNeeded);



    firstMetaPrepareKernel << <occData.blockForFirstMetaPass, occData.theadsForFirstMetaPass >> > (fbArgs, fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer);



    void* kernel_args[] = { &fbArgs };
    cudaLaunchCooperativeKernel((void*)(mainPassKernel<int>), occData.blockForMainPass, dim3(32, occData.warpsNumbForMainPass), kernel_args);


    //copy to the output tensor the rsult
    if (resIterNeeded) {
        auto size = sizeof(int32_t) * fbArgs.resultListPointerIterNumbTensor.sizes()[0];
        cudaMemcpyAsync(fbArgs.resultListPointerIterNumbTensor.data_ptr(), fbArgs.resultListPointerIterNumb, size, cudaMemcpyDeviceToDevice, stream);
    }


    cudaFreeAsync(fbArgs.resultListPointerMeta, stream);
    cudaFreeAsync(fbArgs.resultListPointerLocal, stream);
    cudaFreeAsync(fbArgs.resultListPointerIterNumb, stream);
    cudaFreeAsync(fbArgs.workQueuePointer, stream);
    cudaFreeAsync(fbArgs.origArrsPointer, stream);
    cudaFreeAsync(fbArgs.metaDataArrPointer, stream);
    cudaFreeAsync(fbArgs.mainArrAPointer, stream);
    cudaFreeAsync(fbArgs.mainArrBPointer, stream);


    return fbArgs;

}



#pragma once
template <typename T>
ForBoolKernelArgs<T> mainKernelsRun(ForFullBoolPrepArgs<T>& fFArgs, const int WIDTH, const int HEIGHT, const int DEPTH, cudaStream_t stream, bool resToSave = false,bool  resIterNeeded=false
) {

    //cudaDeviceReset();
    cudaError_t syncErr;
    cudaError_t asyncErr;

    occupancyCalcData occData = getOccupancy<T>();

    //pointers ...
    ForBoolKernelArgs<T> fbArgs = executeHausdoff(fFArgs, WIDTH, HEIGHT, DEPTH, occData, resToSave, stream,1.0, resIterNeeded);

    checkCuda(cudaDeviceSynchronize(), "last ");

    /////////// error handling
    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));


    cudaDeviceReset();

    return fbArgs;
}




//    modified  from https://github.com/pytorch/pytorch/blob/61d6c4386459441710fb4cfa2929a3f77e95e5f7/aten/src/ATen/Dispatch.h
#define AT_DISPATCH_ALL_TYPESWithBool(TYPE, NAME, ...)                               \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                 \
    switch (_st) {                                                              \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Bool, bool, __VA_ARGS__)   \
      default:                                                                  \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");          \
    }                                                                           \
  }()





template <typename T>
int getHausdorffDistance_CUDA_Generic(at::Tensor goldStandard,
    at::Tensor algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, bool resIterNeeded, at::Tensor numberToLookFor) {
    //TODO() use https ://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    MetaDataCPU metaData;
    size_t size = sizeof(unsigned int) * 20;
    unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<T> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = numberToLookFor.item<T>();
    forFullBoolPrepArgs.goldArr = goldStandard;
    forFullBoolPrepArgs.segmArr = algoOutput;

    occupancyCalcData occData = getOccupancy<T>();

    ForBoolKernelArgs<T> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData, stream1, false, robustnessPercent, resIterNeeded);

    size_t sizeMinMax = sizeof(unsigned int) * 20;
    //making sure we have all resultsto copy on cpu
    cudaDeviceSynchronize();
    cudaMemcpy(minMaxesCPU, fbArgs.metaData.minMaxes, sizeMinMax, cudaMemcpyDeviceToHost);

    int result = minMaxesCPU[13];

    cudaFreeAsync(fbArgs.minMaxes, stream1);
    free(minMaxesCPU);


    cudaStreamDestroy(stream1);
    cudaDeviceSynchronize();

    return result;
}


template <typename T>
at::Tensor getHausdorffDistance_CUDA_FullResList_local(at::Tensor goldStandard,
    at::Tensor algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, at::Tensor numberToLookFor) {
    //TODO() use https ://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    MetaDataCPU metaData;
    size_t size = sizeof(unsigned int) * 20;
    unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<T> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = numberToLookFor.item<T>();
    forFullBoolPrepArgs.goldArr = goldStandard;
    forFullBoolPrepArgs.segmArr = algoOutput;

    occupancyCalcData occData = getOccupancy<T>();

    ForBoolKernelArgs<T> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData, stream1, false, robustnessPercent, true);




    cudaFreeAsync(fbArgs.minMaxes, stream1);
    free(minMaxesCPU);



    cudaStreamDestroy(stream1);

    return fbArgs.resultListPointerIterNumbTensor;
}


int getHausdorffDistance_CUDA(at::Tensor goldStandard,
    at::Tensor algoOutput
    , const int WIDTH,const  int HEIGHT,const  int DEPTH
    ,const float robustnessPercent, at::Tensor numberToLookFor) {

    int res = 0;

    AT_DISPATCH_ALL_TYPESWithBool(goldStandard.type(), "getHausdorffDistance_CUDA", ([&] {
        res= getHausdorffDistance_CUDA_Generic<scalar_t>(goldStandard, algoOutput, WIDTH, HEIGHT, DEPTH, robustnessPercent,false, numberToLookFor);

        }));
    return res;
}


at::Tensor getHausdorffDistance_CUDA_FullResList(at::Tensor goldStandard,
    at::Tensor algoOutput
    , const int WIDTH, const  int HEIGHT, const  int DEPTH
    , const float robustnessPercent, at::Tensor numberToLookFor) {


    at::Tensor res;
    AT_DISPATCH_ALL_TYPESWithBool(goldStandard.type(), "getHausdorffDistance_CUDA_FullResList", ([&] {
        res= getHausdorffDistance_CUDA_FullResList_local<scalar_t>(goldStandard, algoOutput, WIDTH, HEIGHT, DEPTH, robustnessPercent, numberToLookFor);

        }));

    return res;
}
