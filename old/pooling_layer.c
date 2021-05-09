#include <stdio.h>
#include <stdlib.h>
#include "util_para.h"

/*
 * Input:  Matrix after ReLU (28x28)
 * Output: Matrix after Pooling (Max or Ave) (28/pooling_window x 28/pooling_window)
 */

void pooling_layer(double **relu_im, double **pooling_im, int pooling_type) {

  double tmp = 0.0;
  double val = 0.0;
  int wi, wj = 0  ; // write back index
  //int testcount = 1; // Delete me

  for (int im_i = 0; im_i < NUM_TRAINS; im_i++) {

    wi = 0;
    for (int bi = 0; bi < IMAGE_DIM; bi = bi + POOLING_WIN_SIZE) { // block index
      wj = 0;
      for (int bj = 0; bj < IMAGE_DIM; bj = bj + POOLING_WIN_SIZE) {
      
        tmp = 0.0;
        for (int ti = 0; ti < POOLING_WIN_SIZE; ti++) {
          for (int tj = 0; tj < POOLING_WIN_SIZE; tj++) {
            val = relu_im[0][(bi+ti)*IMAGE_DIM+bj+tj];
            tmp = (tmp > val) ? tmp : val;
          }
        }

        //printf("pooling_layer.c: tmp is %1.1f , count=%d \n", tmp, testcount);
        //testcount++;  // Delete me
        pooling_im[0][wi*IMAGE_DIM/POOLING_WIN_SIZE+wj] = tmp;

        wj++;
      } // end of for loop of bj
      wi++;
    } // end of for loop of bi

  } // end of for loop of im_i

  printf("~~~~~~~~ pooling_im is ~~~~~~~~ \n");
  for (int i = 0; i < IMAGE_DIM/POOLING_WIN_SIZE * IMAGE_DIM/POOLING_WIN_SIZE; i++) { // TODO: Delete this
    printf("%1.1f ", pooling_im[0][i]);
		if ((i+1) % (IMAGE_DIM/POOLING_WIN_SIZE) == 0) putchar('\n');
  }
  
  printf("This is pooling_layer\n");
}