/*
 *  CUDA for Convolutional Neural Network 
 *  
 *  by Zikun Xu, Bangqi Xiao
 */
 #define USE_MNIST_LOADER
 #define MNIST_DOUBLE
 #include "mnist.h"

 #include <cuda.h>
 #include <cstdio>
 #include <time.h>

 mnist_data *train_set, *test_set; // pointer to dataset struct
 int train_cnt, test_cnt; // sample count

 void load_minist()
 {
	mnist_load("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
 }

 int main(int argc, const char ** argv)
 {
     // CUDA initialisation
     CUresult err = NULL;
     err = cuInit(0);
     if( CUDA_SUCCESS != err ){
         fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
         return 1;
     }
     // loading dataset
     load_minist();

     build_model();
     train();
     test();

     return 0;
 }