#include <stdio.h>
#include <stdlib.h>
#include "util_para.h"
//#include <cuda_runtime.h>
#include <time.h>

#define BILLION 	1000000000

/*
 * Input:  Dataset matrix
 * Output: Result matrix after get convoluted
 */
 void read_data(double **train_im, double **test_im) {

  /*
   * Load MNIST Dataset and store in array
   * - train image : train_image[60000][784] (type: double, normalized, flattened)
   * - train label : train_label[60000]      (type: int)
   * - test image  : test_image[10000][784]  (type: double, normalized, flattened)
   * - test label  : test_label[10000]       (type: int)
   */

  for (int i = 0; i < NUM_TRAINS; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      train_im[i][j] = 1.0;
    }
  }

  for (int i = 0; i < NUM_TESTS; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      test_im[i][j] = 0.0;
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

void padding(double **trained_im, double **padded_im) {

  // ======== 1. Zero-Padding ========

  //puts("test 2 \n");

  int new_dim    = IMAGE_DIM + KERNEL_SIZE - 1;
  int new_size   = new_dim * new_dim;
  int extra_size = KERNEL_SIZE / 2;   // round down automatically

  for (int i = 0; i < NUM_TRAINS; i++) {
    for (int j = 0; j < new_size; j++) {
      //printf("i is %d, j is %d \n", i, j);
      padded_im[i][j] = 0.0;
    }
  }
  //puts("test 3 \n");

  for (int i = 0; i < NUM_TRAINS; i++) {
    for (int j = 0; j < IMAGE_DIM; j++) {
      for (int k = 0; k < IMAGE_DIM; k++) {
        padded_im[i][(j+extra_size)*new_dim+(k+extra_size)] = \
                      trained_im[i][j*IMAGE_DIM+k];
      }
    }
  }

  /*for (int i = 0; i < new_size; i++) { // TODO: Delete this
    printf("%1.1f ", padded_im[0][i]);
		if ((i+1) % 30 == 0) putchar('\n');
  } */

}

void conv_layer_baseline(double **padded_im, double **result_im, int kn_type) {


  int new_dim    = IMAGE_DIM + KERNEL_SIZE - 1;
  int new_size   = new_dim * new_dim;
  int extra_size = KERNEL_SIZE / 2;   // round down automatically
  // ======== 2. Do Convolution Operation  ========

  // -------- 2.1 Create Kernel --------
  double **convKernel = (double **) malloc( KERNEL_SIZE * sizeof(double *));
  for (int i = 0; i < KERNEL_SIZE; i++) {
    convKernel[i] = (double *) malloc(KERNEL_SIZE * sizeof(double));
  }

  if (kn_type == KN_HORIZONTAL_EDGE) {
    convKernel[0][0] = 1;
    convKernel[0][1] = 1;
    convKernel[0][2] = 1;
    convKernel[1][0] = 1;
    convKernel[1][1] = 1;
    convKernel[1][2] = 1;
    convKernel[2][0] = convKernel[0][0] * (1);
    convKernel[2][1] = convKernel[0][1] * (1);
    convKernel[2][2] = convKernel[0][2] * (1);
  }
  else if (kn_type == KN_VERTICAL_EDGE) {
    convKernel[0][0] = -1.0;
    convKernel[1][0] = -2.0;
    convKernel[2][0] = -1.0;
    convKernel[0][1] = 0.0;
    convKernel[1][1] = 0.0;
    convKernel[2][1] = 0.0;
    convKernel[0][2] = convKernel[0][0] * (-1);
    convKernel[1][2] = convKernel[1][0] * (-1);
    convKernel[2][2] = convKernel[2][0] * (-1);
  }
  else {
    printf("conv_layer.c: please set up the kernel \n");
    printf("Usage: Set: KN_VERTICAL_EDGE    to use Kernel to detect vertical   edge \n");
    printf("       Set: KN_HORIZAONTAL_EDGE to use Kernel to detect horizontal edge \n");
    exit(0);
  }

  /*for (int i = 0; i < KERNEL_SIZE; i++) { // TODO: Delete this
    for (int j = 0; j < KERNEL_SIZE; j++) {
      printf("%1.1f ", convKernel[i][j]);
    }
    printf("\n");
  } */

  // -------- 2.2 Load Mapped Image from the dataset --------
  double **imageMap = (double **) malloc( KERNEL_SIZE * sizeof(double *));
  for (int i = 0; i < KERNEL_SIZE; i++) {
    imageMap[i] = (double *) malloc(KERNEL_SIZE * sizeof(double));
  }

  double accu = 0.0;
  int img_i;
  for (img_i = 0; img_i < NUM_TRAINS; img_i++) {

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

      } // end of conv_j loop

    } // end of conv_i loop

  } // end of img_i loop

  printf("test: im_i is %d \n", img_i);

  free(convKernel); free(imageMap);
  //printf("conv_layer: This is conv layer \n");
}

__global__ void conv_cuda(double *dev_padded_im, double *dev_kernel, double *dev_conv_res, int new_dim)
{

  int ki, kj;
  double partial = 0.0;

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  //printf("i = %d, j = %d \n", i, j);
  //if (i > (new_dim-KERNEL_SIZE) || j > (new_dim-KERNEL_SIZE)) {
  //  printf("here 1 \n");
  //}
  //else {
    for (ki = 0; ki < KERNEL_SIZE; ki++) {
      for (kj = 0; kj < KERNEL_SIZE; kj++) {
        partial += dev_padded_im[i*new_dim+j+ki*new_dim+kj] * dev_kernel[ki*KERNEL_SIZE+kj];
      }
    }  //end of ki for loop
    dev_conv_res[i*IMAGE_DIM+j] = partial;
    //printf("here 2 \n");
  //} //end of else
  //printf("test1 \n");
}

__global__ void myKernel()
{
  printf("Hello, world from the device!\n");
}

int main () {

  struct timespec start, end;
  double diff;
  cudaEvent_t start_cu, stop_cu;
  float time;

  double **train_image = (double **) malloc(NUM_TRAINS * sizeof(double*));
  for (int i = 0; i < NUM_TRAINS; i++) {
    train_image[i] = (double *) malloc(IMAGE_SIZE * sizeof(double));
  }

  double **test_image = (double **) malloc(NUM_TESTS * sizeof(double*));
  for (int i = 0; i < NUM_TESTS; i++) {
    test_image[i] = (double *) malloc(IMAGE_SIZE * sizeof(double));
  }

  read_data(train_image, test_image);

  int new_dim    = IMAGE_DIM + KERNEL_SIZE - 1;
  int new_size   = new_dim * new_dim;
  int extra_size = KERNEL_SIZE / 2;   // round down automatically

  double **padded_im = (double **) malloc(NUM_TRAINS * sizeof(double *));
  for (int i = 0; i < NUM_TRAINS; i++) {
    padded_im[i] = (double *) malloc(new_size * sizeof(double));
  }

  double **conv_res = (double **) malloc(NUM_TRAINS * sizeof(double *));
  for (int i = 0; i < NUM_TRAINS; i++) {
    conv_res[i] = (double *) malloc(IMAGE_SIZE * sizeof(double));
  }

  // Baseline part
  clock_gettime(CLOCK_MONOTONIC, &start);
  padding(train_image, padded_im);
  conv_layer_baseline(padded_im, conv_res, KN_HORIZONTAL_EDGE);

  clock_gettime(CLOCK_MONOTONIC, &end);
  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("Baseline conv time is %3.12f \n", diff);
  free(train_image);

  printf(" ======== Baseline Result ======== \n");
  for (int i = 0; i < IMAGE_SIZE; i++) { // TODO: Delete this
    printf("%1.1f ", conv_res[0][i]);
		if ((i+1) % IMAGE_DIM == 0) putchar('\n');
  }

  // Alternative 1
  int size_double_padded_im = NUM_TRAINS * new_size * sizeof(double);
  int size_double_kernel    = KERNEL_SIZE * KERNEL_SIZE * sizeof(double);
  int size_double_conv_res  = NUM_TRAINS * IMAGE_SIZE * sizeof(double);

  double *padded_im_flat = (double *) malloc(size_double_padded_im);
  for (int i = 0; i < NUM_TRAINS; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      padded_im_flat[i*IMAGE_SIZE+j] = padded_im[i][j];
    }
  }
  puts("test 0 \n");

  double *kernel = (double *) malloc(size_double_kernel);
  for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) kernel[i] = 1.0;

  double *h_conv_res = (double *) malloc(size_double_conv_res);
  for (int i = 0; i < NUM_TRAINS * IMAGE_SIZE; i++) h_conv_res[i] = 0.0;

  double *dev_padded_im = NULL, *dev_kernel = NULL, *dev_conv_res = NULL;
  cudaMalloc((void **)&dev_padded_im, size_double_padded_im);
  cudaMalloc((void **)&dev_kernel,    size_double_kernel   );
  cudaMalloc((void **)&dev_conv_res,  size_double_conv_res );

  cudaEventCreate( &start_cu ); cudaEventCreate( &stop_cu );

  //cudaEventRecord(start_cu, 0);

  cudaMemcpy(dev_padded_im, padded_im_flat, size_double_padded_im, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_kernel,    kernel,    size_double_kernel,    cudaMemcpyHostToDevice);
  dim3 Block_A1(15, 15);
  dim3 Grid_A1(new_dim/Block_A1.x, new_dim/Block_A1.y);

  //cudaEventRecord(start_cu, 0);
  float time_acc = 0;
  for (int i = 0; i < NUM_TRAINS; i++) {
    cudaEventRecord(start_cu, 0);
    conv_cuda <<< Grid_A1, Block_A1 >>> (dev_padded_im, dev_kernel, dev_conv_res, new_dim);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_cu, 0);
    cudaEventSynchronize(stop_cu);
    cudaEventElapsedTime(&time, start_cu, stop_cu);
    time_acc += time/1000;
    //printf("time_acc is %3.12f \n", time_acc);
  }
  //cudaDeviceSynchronize();
  //cudaEventRecord(stop_cu, 0);
  //cudaEventSynchronize(stop_cu);
  //cudaEventElapsedTime(&time, start_cu, stop_cu);
  //time = time/1000;
  printf("cuda time is %3.12f \n", time_acc);

  cudaMemcpy(h_conv_res, dev_conv_res, size_double_conv_res, cudaMemcpyDeviceToHost);

  //cudaEventRecord(stop_cu, 0);
  //cudaEventSynchronize(stop_cu);



  cudaEventDestroy(start_cu); cudaEventDestroy(stop_cu);

  cudaFree(dev_padded_im); cudaFree(dev_kernel); cudaFree(dev_conv_res);

  for (int i = 0; i < NUM_TRAINS; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      conv_res[i][j] = h_conv_res[i*IMAGE_SIZE+j];
    }
  }



  printf("end \n");

  myKernel<<<1,10>>>();
  cudaDeviceSynchronize();

  return 0;

}
