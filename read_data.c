#include "mnist.h"

void read_data(double **train_im, double **test_im) {

  /*
   * Load MNIST Dataset and store in array
   * - train image : train_image[60000][784] (type: double, normalized, flattened)
   * - train label : train_label[60000]      (type: int)
   * - test image  : test_image[10000][784]  (type: double, normalized, flattened)
   * - test label  : test_label[10000]       (type: int)
   */
  load_mnist();

  train_im = (double **)train_image;
  test_im  = (double **)test_image;

  printf("read_data.c: Loaded images to arrays \n");

  //save_mnist_pgm(test_im, 0);

}