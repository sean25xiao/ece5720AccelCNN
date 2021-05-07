// Declare all the functions here

//#define IMAGE_SIZE 784 // 28*28=784

void read_data(double **train_im, double **test_im);
void conv_layer(double **trained_im, double **result_im, int kn_type);
void relu_layer(double **conv_im);
void pooling_layer(double **relu_im, double **pooling_im, int pooling_type);