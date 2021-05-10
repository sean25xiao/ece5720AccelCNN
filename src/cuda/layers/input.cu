#include "input.h"

Input::Input(int width, int height){
    this->O = width*height;
    cudaMalloc(&output, sizeof(float) * width* height);
}

Input::~Input(){
    cudaFree(output);
}

void Input::forward(float* data){
    cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

