// Declare all the functions here

//#define IMAGE_SIZE 784 // 28*28=784

void read_data(double **train_im, double **test_im);
void conv_layer(double **trained_im, double **result_im);
void relu_layer(double **conv_im);