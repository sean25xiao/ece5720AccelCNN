#include "model.h"

Model::Model(float learning_rate){
    this->learning_rate = learning_rate;
	cublasCreate(&handle);
};

Model::~Model(){}

// Feed one sample
float Model::feed(float data[28][28], int label, bool isTrain){
	// should timing here
	forward(data);
	conv_2.backward_reset();

	Loss::softmax(conv_2.d_output, conv_2.output, label, 10);

	float err;
	cublasSnrm2(handle, 10, conv_2.d_output, 1, &err);

	if(isTrain)   
		backward();
	
	return err;
};

// Forward
void Model::forward(float data[28][28]){
	conv_1.forward_reset();
	conv_2.forward_reset();

    input_layer.forward(*data);
    conv_1.forward(input_layer.output);
    conv_2.forward(conv_1.output);
};

// Backward 
void Model::backward(){
	conv_1.backward_reset();

    conv_2.backward(conv_1.output, conv_1.d_output);
    conv_1.backward(input_layer.output, NULL);

	apply_grad(conv_1.weight, conv_1.d_weight, conv_1.weight_dim);
    apply_grad(conv_2.weight, conv_2.d_weight, conv_2.weight_dim);
};

void Model::apply_grad(float* weight, float* d_weight, int N)
{
    cuda_apply_grad<<<64,64>>>(weight, d_weight, N, learning_rate);
}

__global__ void cuda_apply_grad(float* weight, float* d_weight, int N, float learning_rate)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		weight[idx] += learning_rate * d_weight[idx];
	}
}

// Predict the class of input sample
int Model::predict(float data[28][28])
{
	forward(data);

	int predLabel = 0;
	float outputVec[10];

	cudaMemcpy(outputVec, conv_2.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (outputVec[predLabel] < outputVec[i]) {
			predLabel = i;
		}
	}

	return predLabel;
}
