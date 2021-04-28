#include <stdio.h>
#include <stdlib.h>
#include "util_para.h"

/*
 * Input:  Dataset matrix
 * Output: Result matrix after get convoluted
 */

void conv_layer(double **trained_im, double **result_im) {


  // ======== 1. Zero-Padding ========
  int new_dim    = IMAGE_DIM + KERNEL_SIZE - 1;
  int new_size   = new_dim * new_dim;
  int extra_size = KERNEL_SIZE / 2;   // round down automatically
  double **padded_im = (double **) malloc(NUM_TRAINS * sizeof(double *));
  for (int i = 0; i < NUM_TRAINS; i++) {
    padded_im[i] = (double *) malloc(new_size * sizeof(double));
  }
  puts("test 2 \n");

  for (int i = 0; i < NUM_TRAINS; i++) {
    for (int j = 0; j < new_size; j++) {
      //printf("i is %d, j is %d \n", i, j);
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

  for (int i = 0; i < new_size; i++) { // TODO: Delete this
    printf("%1.1f ", padded_im[0][i]);
		if ((i+1) % 30 == 0) putchar('\n');
  }

  // ======== 2. Do Convolution Operation  ========

  // -------- 2.1 Create Kernel --------
  double **convKernel = (double **) malloc( KERNEL_SIZE * sizeof(double *));
  for (int i = 0; i < KERNEL_SIZE; i++) {
    convKernel[i] = (double *) malloc(KERNEL_SIZE * sizeof(double));
  }

  for (int i = 0; i < KERNEL_SIZE; i++) {
    for (int j = 0; j < KERNEL_SIZE; j++) {
      convKernel[i][j] = -1.0;
    }
  }
  convKernel[KERNEL_SIZE/2][KERNEL_SIZE/2] = (double)KERNEL_CENTER;
  
  for (int i = 0; i < KERNEL_SIZE; i++) { // TODO: Delete this
    for (int j = 0; j < KERNEL_SIZE; j++) {
      printf("%1.1f ", convKernel[i][j]);
    }
    printf("\n");
  }

  // -------- 2.2 Load Mapped Image from the dataset --------
  double **imageMap = (double **) malloc( KERNEL_SIZE * sizeof(double *));
  for (int i = 0; i < KERNEL_SIZE; i++) {
    imageMap[i] = (double *) malloc(KERNEL_SIZE * sizeof(double));
  }

  double accu = 0.0;
  for (int img_i = 0; img_i < NUM_TRAINS; img_i++) {

    /// conv_i and conv_j are the top left element of kernel
    for (int conv_i = 0; conv_i <= new_dim-KERNEL_SIZE; conv_i = conv_i + CONV_STRIDE) {
      for (int conv_j = 0; conv_j <= new_dim-KERNEL_SIZE; conv_j = conv_j + CONV_STRIDE) {
      
        accu = 0.0;

        // Load Mapped Image to imageMap
        // Then imageMap conv with convKernel
        for (int i = 0; i < KERNEL_SIZE; i++) {
          for (int j = 0; j < KERNEL_SIZE; j++) {
            imageMap[i][j] = padded_im[img_i][(i+conv_i)*new_dim+(j+conv_j)];
            accu += imageMap[i][j] * convKernel[i][j];
          }
        }

        result_im[img_i][conv_i*IMAGE_DIM+conv_j] = accu + CONV_BIAS;

        /*printf("conv_i is %d, conv_j is %d \n", conv_i, conv_j);
        for (int i = 0; i < KERNEL_SIZE; i++) { // TODO: Delete this
          for (int j = 0; j < KERNEL_SIZE; j++) {
            printf("%1.1f ", imageMap[i][j]);
          }
          printf("\n");
        }
        printf("conv accu is %1.1f \n", accu); */

      } // end of conv_j loop

    } // end of conv_i loop

  } // end of img_i loop

  free(padded_im); free(convKernel); free(imageMap);
  printf("conv_layer: This is conv layer \n");
}