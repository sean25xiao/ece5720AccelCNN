/*
 * Convolutional Layer
 */
#define LEARNING_RATE 0.01

class Conv2d
{
private:
    int in_w; 
    int in_h;
    int in_c;

    int k_sz;

    int ot_w;
    int ot_h;
    int ot_c;

    int stride;

    int M;
    int N;
    int O;

public:
    int weight_dim;
    float* prev_output;
    float* prev_d_output;

    float* output;
    float* weight;
    float* bias;

    float* d_output;
    float* d_weight;

    Conv2d(int input_width, int input_height, int input_channels, 
        int kernel_size, int output_channels, int stride);
    ~Conv2d();

    void forward(float* prev_output);
    void backward(float* prev_output, float* prev_d_output);

    void forward_reset();
    void backward_reset();
};

__global__ void forward_conv(float* input, float* output, float* weight, 
        int k_sz, int ot_w, int ot_h, int ot_c);

__global__ void forward_add_bias(float *output, float *bias, int ot_w, int ot_h, int ot_c);


__global__ void backward_gradient_weight(float *d_weight, float *d_output, float *prev_output, int k_sz, int ot_w, int ot_h, int ot_c);


__global__ void backward_update_bias(float *bias, float *d_output, int ot_w, int ot_h, int ot_c);

__global__ void backward_gradient_prev(float *prev_d_output, float *weight, float* d_output, int k_sz, int ot_w, int ot_h, int ot_c);

