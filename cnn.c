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

  double **conv_res = (double **) malloc(NUM_TRAINS * sizeof(double *));
  for (int i = 0; i < NUM_TRAINS; i++) {
    conv_res[i] = (double *) malloc(IMAGE_SIZE * sizeof(double));
  }

  conv_layer(train_image, conv_res);

  relu_layer(conv_res);

  free(train_image); free(test_image); free(conv_res);

  return 0;
}