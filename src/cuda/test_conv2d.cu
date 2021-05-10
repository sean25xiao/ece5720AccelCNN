#include <cuda.h>
#include <cstdio.h>
#include "layers/input.h"
#include "layers/conv2d.h"

void printTestFunc(float* data, int N){
    float h_data[N];

    for (int i=0; i<N; i++){
        h_data[i] = 0;
    }
    cudaMemcpy(h_data, data, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i=0; i<N; i++){
        printf("%d ", h_data[i]);
    }
}

int main()
{
    float data[28][28];

    for(int i=0; i<28*28; i++){
        for(int j=0; j<28; j++){
            data[i][j] = 1;
        }
    }

    Input input_layer(28,28);
    input_layer.forward((float *)data);
    printTestFunc(input_layer.output)
}

