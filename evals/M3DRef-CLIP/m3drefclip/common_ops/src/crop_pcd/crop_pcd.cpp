//
// Created by Yiming Zhang on 4/26/23.
//

#include "crop_pcd.h"

void cropPcdsFromAabbs(const at::Tensor &aabbMinMaxBoundsTensor, const at::Tensor &pcdsTensor, const at::Tensor &outputMasksTensor) {
    const float *pcds = pcdsTensor.data_ptr<float>();
    const float *aabbMinMaxBounds = aabbMinMaxBoundsTensor.data_ptr<float>();
    bool *outputMasks = outputMasksTensor.data_ptr<bool>();

    const int numAabbs = aabbMinMaxBoundsTensor.sizes()[0];
    const int numChunks = pcdsTensor.sizes()[0];

    cropPcdsFromAabbsCuda(aabbMinMaxBounds, pcds, outputMasks, numAabbs, numChunks);
}