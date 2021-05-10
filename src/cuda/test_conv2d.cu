#include <cstdio>
#include "model.h"

void printTestFunc(float* data, int a, int b, int c){
    int N = a*b*c;
    float h_data[N];

    for (int i=0; i<N; i++){
        h_data[i] = 0;
    }
    cudaMemcpy(h_data, data, sizeof(float) * N, cudaMemcpyDeviceToHost);

    for (int i=0; i<a; i++){
        for(int j=0;j<b;j++){
            for(int k=0; k<c; k++){
                printf("%f ", h_data[i*b*c+j*c+k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

void anotherTest(float* A){

    printf("%f \n",*(A+6));
}

int main()
{
    int M=16;
    float data[M][M];

    for(int i=0; i<M; i++){
        for(int j=0; j<M; j++){
            data[i][j] = 1;
        }
    }

    // forward 
    Input input_layer(M,M);
    input_layer.forward(*data);
    
    
    Conv2d conv_1(M,M,1,5,6,1); // 6,6,6
    conv_1.forward_reset();
    conv_1.forward(input_layer.output);

    Conv2d conv_2(12,12,1,12,10,1);
    conv_2.forward_reset();
    conv_2.forward(conv_1.output);

    // printTestFunc(conv_2.output, 1,1,10);

    conv_2.backward_reset();
    conv_1.backward_reset();

    Loss::softmax(conv_2.d_output, conv_2.output, 0, 10);
    conv_2.backward(conv_1.output, conv_1.d_output);

    conv_1.backward(input_layer.output, NULL);

    Model model(0.01);
    model.apply_grad(conv_1.weight, conv_1.d_weight, conv_1.weight_dim);
    model.apply_grad(conv_2.weight, conv_2.d_weight, conv_2.weight_dim);

    
    // printTestFunc(conv_2.d_output, 1,1,10);




    // Conv2d conv_2(12,12,6,12,1,1);
    // conv_2.forward_reset();
    // conv_2.forward(conv_1.output);


    
}

