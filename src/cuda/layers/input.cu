#include "input.h"
#include <cstdio>

Input::Input(int width, int height){
    this->O = width*height;
    cudaMalloc(&output, sizeof(float) * width* height);
    printf("Input Layer, output_shape=(%d,%d)\n", width, height);
}

Input::~Input(){
    cudaFree(output);
}

void Input::forward(float* data){
    cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

