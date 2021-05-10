#include "conv2d.h"
#include <cstdio>

Conv2d::Conv2d(int input_width, int input_height, int input_channels, int kernel_size, int output_channels, int stride)
{
    // assign input shape
    this->in_w = input_width;
    this->in_h = input_height;
    this->in_c = input_channels;

    // assign kernel shape and stride
    this->k_sz = kernel_size;
    this->stride = stride;

    // output shape
    this->ot_w = (in_w - kernel_size) / stride + 1;
    this->ot_h = (in_h - kernel_size) / stride + 1;
    this->ot_c = output_channels;

    printf("Conv Layer, input_shape=(%d,%d,%d), output_shape=(%d,%d,%d)\n",in_c, in_w, in_h, ot_c, ot_w, ot_h);

    this->M = k_sz * k_sz;
    this->N = ot_c;
    this->O = ot_w * ot_h * ot_c;
    this->weight_dim = N*M;
    // random initialize weight and bias
    float h_weight[N][M];
    float h_bias[N];

    for (int i = 0; i < N; ++i)
    {
        h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
        // h_bias[i] = 0; // DEBUG

        for (int j = 0; j < M; ++j)
        {
            h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
            // h_weight[i][j] = 1; // DEBUG
        }
    }

    // allocate cuda memory
    cudaMalloc(&output, sizeof(float) * O);
    cudaMalloc(&weight, sizeof(float) * M * N);
    cudaMalloc(&bias, sizeof(float) * N);

    cudaMalloc(&d_output, sizeof(float) * O);
    cudaMalloc(&d_weight, sizeof(float) * M * N);

    // copy initialized weight and biase from host memory to device memory
    cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

Conv2d::~Conv2d()
{
    // Free cuda memory
    cudaFree(output);
    cudaFree(weight);
    cudaFree(bias);

    cudaFree(d_output);
    cudaFree(d_weight);
}

// Memory clear function used between iterations
void Conv2d::forward_reset()
{
    cudaMemset(output, 0x00, sizeof(float) * O);
}

void Conv2d::backward_reset()
{
    cudaMemset(d_output, 0x00, sizeof(float) * O);
    cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}

/* 
 *   Forward pass of Convolutional Layer, 
 *   IMPORTANT: prev_output should be the output of last layer 
 */
void Conv2d::forward(float *prev_output)
{
    // Convolution
    forward_conv<<<64, 64>>>(prev_output, output, weight, k_sz, ot_w, ot_h, ot_c);

    // Add bias
    forward_add_bias<<<64, 64>>>(output, bias, ot_w, ot_h, ot_c);
};

/*
 *  Backward pass of Convolutional Layer,
 *  IMPORTANT: make sure d_output of current layer is already assigned before run this function
 */
void Conv2d::backward(float *prev_output, float *prev_d_output)
{
    // Compute gradient of weight
    backward_gradient_weight<<<64, 64>>>(d_weight, d_output, prev_output, k_sz, ot_w, ot_h, ot_c);
    // Update Bias
    backward_update_bias<<<64, 64>>>(bias, d_output, ot_w, ot_h, ot_c);
    // Update prev_d_output
    if ( prev_d_output == NULL) return;
    backward_gradient_prev<<<64, 64>>>(prev_d_output, weight, d_output, k_sz, ot_w, ot_h, ot_c);
};

__global__ void backward_gradient_weight(float *d_weight, float *d_output, float *prev_output,
                                         int k_sz, int ot_w, int ot_h, int ot_c)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = ot_c * k_sz * k_sz * ot_w * ot_h;

    float d = pow(24.0f, 2.0f);

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % ot_c);
        int i2 = ((idx /= ot_c) % k_sz);
        int i3 = ((idx /= k_sz) % k_sz);
        int i4 = ((idx /= k_sz) % ot_w);
        int i5 = ((idx /= ot_w) % ot_h);

        atomicAdd(&d_weight[i1 * k_sz * k_sz + i2 * k_sz + i3], d_output[i1 * ot_w * ot_h + i4 * ot_h + i5] * prev_output[(i4 + i2) * (k_sz + ot_h) + i5 + i3] / d);
        // atomicAdd(&d_weight[i1][i2][i3], d_output[i1][i4][i5] * prev_output[i4 + i2][i5 + i3] / d);
    }
}

__global__ void backward_update_bias(float *bias, float *d_output,
                                     int ot_w, int ot_h, int ot_c)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = ot_c * ot_w * ot_h;
    float d = pow(24.0f, 2.0f);

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % ot_c);
        int i2 = ((idx /= ot_c) % ot_w);
        int i3 = ((idx /= ot_w) % ot_h);

        atomicAdd(&bias[i1], LEARNING_RATE * d_output[i1 * ot_w * ot_h + i2 * ot_h + i3] / d);
        // atomicAdd(&bias[i1], LEARNING_RATE * d_output[i1][i2][i3] / d);
    }
}

__global__ void backward_gradient_prev(float *prev_d_output, float *weight, float *d_output,
                                       int k_sz, int ot_w, int ot_h, int ot_c)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = k_sz * k_sz * ot_c * ot_w * ot_h;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 1);
        int i2 = ((idx /= 1) % k_sz);
        int i3 = ((idx /= k_sz) % k_sz);
        int i4 = ((idx /= k_sz) % ot_c); // ?
        int i5 = ((idx /= ot_c) % ot_w);
        int i6 = ((idx /= ot_w) % ot_h);

        atomicAdd(&prev_d_output[i4 * (ot_w * 4 + k_sz) + (i5 * 4 + i2) * (ot_h * 4 + k_sz) + i6 * 4 + i3],
                  weight[i1 * k_sz * k_sz + i2 * k_sz + i3] * d_output[i4 * ot_w * ot_h + i5 * ot_h + i6]);
        // atomicAdd(&prev_d_output[i4][i5 * 4 + i2][i6 * 4 + i3], weight[i1][i2][i3] * d_output[i4][i5][i6]);
    }
}

__global__ void forward_conv(float *input, float *output, float *weight,
                             int k_sz, int ot_w, int ot_h, int ot_c)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = k_sz * k_sz * ot_c * ot_w * ot_h;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % k_sz);
        int i2 = ((idx /= k_sz) % k_sz);
        int i3 = ((idx /= k_sz) % ot_c);
        int i4 = ((idx /= ot_c) % ot_w);
        int i5 = ((idx /= ot_w) % ot_h);

        atomicAdd(&output[i3*ot_w*ot_h+i4*ot_h+i5], weight[i3*k_sz*k_sz+i1*k_sz + i2] * input[(i4 + i1)*(k_sz+ot_h)+i5 + i2]);
        // atomicAdd(&output[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
    }
}

__global__ void forward_add_bias(float *output, float *bias, int ot_w, int ot_h, int ot_c)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = ot_c * ot_w * ot_h;

    for (int n = N * pos / size; n < N * (pos + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % ot_c);
        int i2 = ((idx /= ot_c) % ot_w);
        int i3 = ((idx /= ot_w) % ot_h);

        output[i1 *ot_w*ot_h+i2*ot_h+i3] += bias[i1];
        // output[i1][i2][i3] += bias[i1];
    }
}
