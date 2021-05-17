#include "common.h"

/** 
 * @brief sample from a uniform distribution
 */
float nn::generateUniformRandom(float range)
{
    return range - float(rand()) / float(RAND_MAX);
}

void nn::apply_grad(float* param, float* grad, float lr, int param_fdim)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);

    float nlr = -lr;

    stat = cublasSaxpy(handle, param_fdim, &nlr, grad, 1, param, 1);
    if(stat != CUBLAS_STATUS_SUCCESS)
    {
        printf("cublasSaxpy failed\n");
    }
}
