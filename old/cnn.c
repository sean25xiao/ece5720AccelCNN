#include <stdio.h>
#include <stdlib.h>
#include "util_func.h"
#include "util_para.h"
//#include "mnist.h"

//#define IMAGE_SIZE 784 // 28*28=784
//#define NUM_TRAINS 60000
//#define NUM_TESTS 10000

int main () {

  double **train_image = (double **) malloc(NUM_TRAINS * sizeof(double*));
  for (int i = 0; i < NUM_TRAINS; i++) {
    train_image[i] = (double *) malloc(IMAGE_SIZE * sizeof(double));
  }

  double **test_image = (double **) malloc(NUM_TESTS * sizeof(double*));
  for (int i = 0; i < NUM_TESTS; i++) {
    test_image[i] = (double *) malloc(IMAGE_SIZE * sizeof(double));
  }
 
  read_data(train_image, test_image);

  /*for (int i = 0; i < IMAGE_SIZE; i++) {
    printf("%1.1f ", test_image[0][i]);
    if ((i+1) % 28 == 0) putchar('\n');
  } */

  // ========================================================
 
  // Convolution 1.0
  double **conv_1 = (double **) malloc(NUM_TRAINS * sizeof(double *));
  for (int i = 0; i < NUM_TRAINS; i++) {
    conv_1[i] = (double *) malloc(IMAGE_SIZE * sizeof(double));
  }
  conv_layer(train_image, conv_1, KN_HORIZONTAL_EDGE);
  free(train_image); free(test_image);

  // ReLU 1.0
  relu_layer(conv_1);

  // Convolution 2.0
  /*double **conv_2 = (double **) malloc(NUM_TRAINS * sizeof(double *));
  for (int i = 0; i < NUM_TRAINS; i++) {
    conv_2[i] = (double *) malloc(IMAGE_SIZE * sizeof(double));
  }
  conv_layer(conv_1, conv_2, KN_VERTICAL_EDGE);
  free(conv_1); */

  // Pooling 1.0
  double **pooling_res = (double **) malloc(NUM_TRAINS * sizeof(double *));
  for (int i = 0; i < NUM_TRAINS; i++) {
    pooling_res[i] = (double *) malloc((IMAGE_DIM/POOLING_WIN_SIZE) * (IMAGE_DIM/POOLING_WIN_SIZE) * sizeof(double));
  }
  //printf("p_res[0][0] is %1.1f \n", pooling_res[0][0]);
  pooling_layer(conv_1, pooling_res, 1);

  free(conv_1); free(pooling_res);

  return 0;
}