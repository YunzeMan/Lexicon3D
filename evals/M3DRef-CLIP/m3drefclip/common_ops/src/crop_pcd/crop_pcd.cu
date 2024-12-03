#include "crop_pcd.h"

__global__ void cropPcdsFromAabbsCudaKernel(const float *aabbMinMaxBounds, const float *pcds, bool *outputMasks) {

    __shared__ float sharedMemCurrentPoint[3];

    const int pcdStartIndex = blockIdx.x * 3;

    if (threadIdx.x < 3) {
        sharedMemCurrentPoint[threadIdx.x] = pcds[pcdStartIndex + threadIdx.x];
    }

    __syncthreads();

    const int currentAabbMinIndexStart = threadIdx.x * 6;
    const float &aabbMinX = aabbMinMaxBounds[currentAabbMinIndexStart];
    const float &aabbMinY = aabbMinMaxBounds[currentAabbMinIndexStart + 1];
    const float &aabbMinZ = aabbMinMaxBounds[currentAabbMinIndexStart + 2];
    const float &aabbMaxX = aabbMinMaxBounds[currentAabbMinIndexStart + 3];
    const float &aabbMaxY = aabbMinMaxBounds[currentAabbMinIndexStart + 4];
    const float &aabbMaxZ = aabbMinMaxBounds[currentAabbMinIndexStart + 5];

    const float &pcdX = sharedMemCurrentPoint[0];
    const float &pcdY = sharedMemCurrentPoint[1];
    const float &pcdZ = sharedMemCurrentPoint[2];

    if (pcdX >= aabbMinX && pcdX <= aabbMaxX && pcdY >= aabbMinY && pcdY <= aabbMaxY && pcdZ >= aabbMinZ && pcdZ <= aabbMaxZ) {
        outputMasks[threadIdx.x * gridDim.x + blockIdx.x] = true;
    }
}

void cropPcdsFromAabbsCuda(const float *aabbMinMaxBounds, const float *pcds,
                           bool *outputMasks, const int &numAabbs, const int &numChunks) {
    cropPcdsFromAabbsCudaKernel<<<numChunks, numAabbs, sizeof(float) * 3>>>(aabbMinMaxBounds, pcds, outputMasks);
}