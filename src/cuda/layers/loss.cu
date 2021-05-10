#include "loss.h"

void Loss::softmax(float* loss, float* logit, int label,int numClass)
{
    cudaSoftmax<<<1, numClass>>>(loss, logit, label, numClass);
}

__global__ void cudaSoftmax(float* loss, float* logit, int label,int numClass)
{
    int tid =  threadIdx.x;

	loss[tid] = ((label == tid ? 1.0f : 0.0f) - logit[tid]);
}
