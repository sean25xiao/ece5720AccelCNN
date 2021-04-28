#include <stdio.h>
#include <stdlib.h>
#include "util_para.h"

/*
 * Input:  Dataset matrix
 * Output: Result matrix after get convoluted
 */


void conv_layer(double **trained_im, double **result_im) {

  // ======== Allocate the Convolution Kernel ========
  double **convKernel = (double **) malloc( KERNEL_SIZE * sizeof(double *));
  for (int i = 0; i < KERNEL_SIZE; i++) {
    convKernel[i] = (double *) malloc(KERNEL_SIZE * sizeof(double));
  }

  double **imageMap = (double **) malloc( KERNEL_SIZE * sizeof(double *));
  for (int i = 0; i < KERNEL_SIZE; i++) {
    imageMap[i] = (double *) malloc(KERNEL_SIZE * sizeof(double));
  }
  puts("test 1 \n");

  // ======== 1. Zero-Padding ========
  int new_dim    = IMAGE_DIM + KERNEL_SIZE - 1;
  int new_size   = new_dim * new_dim;
  int extra_size = KERNEL_SIZE / 2;   // round down automatically
  double **padded_im = (double **) malloc(NUM_TRAINS * sizeof(double *));
  for (int i = 0; i < KERNEL_SIZE; i++) {
    padded_im[i] = (double *) malloc(new_size * sizeof(double));
  }
  puts("test 2 \n");

  for (int i = 0; i < NUM_TRAINS; i++) {
    for (int j = 0; j < new_size; j++) {
      printf("i is %d, j is %d \n", i, j);
      padded_im[i][j] = 0.0;
    }
  }
  puts("test 3 \n");

  for (int i = 0; i < NUM_TRAINS; i++) {
    for (int j = 0; j < IMAGE_DIM; j++) {
      for (int k = 0; k < IMAGE_DIM; k++) {
        padded_im[i][(j+extra_size)*new_dim+(k+extra_size)] = \
                      trained_im[i][j*IMAGE_DIM+k];
      }
    }
  }
  puts("test 4 \n");

  for (int i = 0; i < new_dim; i++) {
    printf("%lf, ", padded_im[0][i]);
  }

  // ======== Load the Mapped Image part from trained_im to imageMap ========
  for (int i = 0; i < KERNEL_SIZE; i++) {
    for (int j = 0; j < KERNEL_SIZE; j++) {
      //imageMap[i][j] = trained_im[][];
    }
  }

  free(padded_im); free(convKernel); free(imageMap);
  printf("conv_layer: This is conv layer \n");
}