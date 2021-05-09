#include "mnist.h"
#include "util_para.h"

void read_data(double **train_im, double **test_im) {

  /*
   * Load MNIST Dataset and store in array
   * - train image : train_image[60000][784] (type: double, normalized, flattened)
   * - train label : train_label[60000]      (type: int)
   * - test image  : test_image[10000][784]  (type: double, normalized, flattened)
   * - test label  : test_label[10000]       (type: int)
   */
  load_mnist();

  for (int i = 0; i < NUM_TRAINS; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      train_im[i][j] = train_image[i][j];
    }
  }

  for (int i = 0; i < NUM_TESTS; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      test_im[i][j] = test_image[i][j];
    }
  } 

  printf("read_data.c: Loaded images to arrays \n");

  //save_mnist_pgm(train_im, 0);

  int i;
	for (i=0; i<784; i++) {
    //printf("i is %d \n", i);
		printf("%1.1f ", train_im[0][i]);
		if ((i+1) % 28 == 0) putchar('\n');
	} 

}