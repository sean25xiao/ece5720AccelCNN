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

  puts("test 1 \n");
  for (int bi = 0; bi < IMAGE_DIM; bi = bi + POOLING_WIN_SIZE) { // block index
    for (int bj = 0; bj < IMAGE_DIM; bj = bj + POOLING_WIN_SIZE) {
      
      tmp = 0.0;
      for (int ti = 0; ti < POOLING_WIN_SIZE; ti++) {
        for (int tj = 0; tj < POOLING_WIN_SIZE; tj++) {
          //val = relu_im[bi+ti][bj+tj];
          //puts("test 2 \n");
          val = relu_im[0][(bi+ti)*IMAGE_DIM+bj+tj];
          tmp = (tmp > val) ? tmp : val;
        }
      }
     puts("test 3 \n");
      //pooling_im[wi][wj] = tmp;
      pooling_im[0][wi*POOLING_WIN_SIZE+wj] = tmp;
     puts("test 4 \n");
      wj++;
    }
    wi++;
  }
  puts("test 5 \n");
  
  printf("This is pooling_layer\n");
}