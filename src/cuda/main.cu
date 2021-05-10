/*
 *  CUDA for Convolutional Neural Network 
 *  
 *  by Zikun Xu, Bangqi Xiao
 */
#define USE_MNIST_LOADER
#define MNIST_FLOAT

#include "mnist.h"
#include "model.h"
#include <cuda.h>
#include <cstdio>
#include <time.h>

mnist_data *train_set, *test_set; // pointer to dataset struct
unsigned int train_cnt, test_cnt;          // sample count

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

void load_minist()
{
    printf("Loading data .. \n");
    int ret = mnist_load("../../data/train-images.idx3-ubyte", "../../data/train-labels.idx1-ubyte",
               &train_set, &train_cnt);
    
    mnist_load("../../data/t10k-images.idx3-ubyte", "../../data/t10k-labels.idx1-ubyte",
               &test_set, &test_cnt);
        
    printf("Successfully loaded %d training samples and %d testing samples\n", train_cnt, test_cnt);


}

int main(int argc, const char **argv)
{
    // CUDA initialisation
    CUresult err_code = cuInit(0);
    if (CUDA_SUCCESS != err_code)
    {
        fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err_code);
        return 1;
    }
    // loading dataset
    load_minist();
    // Create model instance
    printf("\n==========================================\n");
    printf("Creating CNN model...\n");
    Model cnn = Model(0.01);
    printf("==========================================\n\n");

    // Start training
    int EPOCH = 20;
    clock_t start, end;
    double timing=0;

    for (int epoch = 1; epoch < EPOCH; epoch++)
    {
        float loss = 0;
        
	    start = clock();
        for (int i = 0; i < train_cnt; i++)
        { 
            loss += cnn.feed(train_set[i].data, train_set[i].label, true);
        }
        end = clock();

        timing += ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Epoch %d, loss=%.3f, time %.3e\n", epoch, loss, timing);        
    }

    return 0;
}
