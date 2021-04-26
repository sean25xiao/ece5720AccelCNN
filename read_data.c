#include "mnist.h"

int read_data(void) {
  // Load MNIST Dataset and store in array
  /*
   * - train image : train_image[60000][784] (type: double, normalized, flattened)
   * - train label : train_label[60000]      (type: int)
   * - test image  : test_image[10000][784]  (type: double, normalized, flattened)
   * - test label  : test_label[10000]       (type: int)
   */
  load_mnist();

  printf("read_data.c: Loaded images to arrays \n");

  return 0;
}