#include <stdio.h>
#include <stdlib.h>
#include "util_para.h"

/*
 * ReLU: Pixel-by-Pixel operation that replaces all negative values by 0
 * Input: Convoluted matrix
 * Output: Rectified Linear Matrix
 */

void relu_layer(double **conv_im) {

  for (int i = 0; i < IMAGE_SIZE; i++) { // TODO: Delete this
    printf("%1.1f ", conv_im[0][i]);
		if ((i+1) % IMAGE_SIZE == 0) putchar('\n');
  }
  printf("=================\n");

  for (int i = 0; i < NUM_TRAINS; i++) {
    for(int j = 0; j < IMAGE_SIZE; j++) {
      if (conv_im[i][j] < 0 )
        conv_im[i][j] = (double)ZERO;
    }
  }

  for (int i = 0; i < IMAGE_SIZE; i++) { // TODO: Delete this
    printf("%1.1f ", conv_im[0][i]);
		if ((i+1) % IMAGE_SIZE == 0) putchar('\n');
  }
  printf("relu_layer: This is ReLU layer \n");
}