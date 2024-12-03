//
// Created by Yiming Zhang on 4/26/23.
//

#ifndef CROP_PCD_H
#define CROP_PCD_H
#include <torch/serialize/tensor.h>

void cropPcdsFromAabbs(const at::Tensor &aabbMinMaxBoundsTensor, const at::Tensor &pcdsTensor, const at::Tensor &outputMasksTensor);

void cropPcdsFromAabbsCuda(const float *aabbMinMaxBounds, const float *pcds,
                           bool *outputMasks, const int &numAabbs, const int &numChunks);

#endif //CROP_PCD_H
