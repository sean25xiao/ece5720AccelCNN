#include "layers/input.h"
#include "layers/conv2d.h"
#include "layers/loss.h"

#include <cublas_v2.h>
#include <cuda.h>

class Model
{
private:
    float learning_rate;
    cublasHandle_t handle;    

    void forward(float data[28][28]);
    void backward();

public:
    // Build model structure here
    Input input_layer = Input(28,28);
    Conv2d conv_1 = Conv2d(28,28,1,5,6,1);
    Conv2d conv_2 = Conv2d(24,24,6,24,10,1);

    Model(float learning_rate);
    ~Model();
    float feed(float data[28][28], int label, bool isTrain);
    int predict(float data[28][28]);
    // void test();
    void apply_grad(float* weight, float* d_weight, int N);
};

__global__ void cuda_apply_grad(float* weight, float* d_weight, int N, float learning_rate);