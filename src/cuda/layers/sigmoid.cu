#include "sigmoid.h"
#include <cstdio>

Sigmoid::Sigmoid(int inputDim)
{
    this->N = inputDim;
    cudaMalloc(&output, sizeof(float) * N);
    cudaMalloc(&d_output, sizeof(float) * N);
    printf("Sigmoid Layer, input_size=%d\n", N);
}

Sigmoid::~Sigmoid()
{
    cudaFree(output);
    cudaFree(d_output);
}

void Sigmoid::forward(float* prev_output)
{
    cudaSigmoidForward<<<64,64>>>(prev_output, output, N);
}

//IMPORTANT: make sure forward output vec is kept before run backward
void Sigmoid::backward(float* prev_d_output)
{
    backward_gradient_prev<<<64,64>>>(prev_d_output, d_output, output, N);
}


void Sigmoid::forward_reset()
{
   cudaMemset(output, 0x00, sizeof(float)*N);
}

void Sigmoid::backward_reset()
{
    cudaMemset(d_output, 0x00, sizeof(float)*N);    
}

__global__ void cudaSigmoidForward(float* prev_output, float* output, int N)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    for (int i = N*pos/size; i<N*(pos+1)/size; ++i)
    {
        output[i] = 1/(1+exp(-prev_output[i]));
    }
}

__global__ void backward_gradient_prev(float* prev_d_output, float* d_output, float* output,int N)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    for (int i = N*pos/size; i<N*(pos+1)/size; ++i)
    {
        prev_d_output[i] = d_output[i]*output[i]*(1-output[i]);
    } 
}