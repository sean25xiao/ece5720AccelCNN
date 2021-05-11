/*
 * Sigmoid Layer
 */
class Sigmoid
{
private:
    int N;
public:
    float* output;
    float* d_output;

    Sigmoid(int inputDim);
    ~Sigmoid();
    void forward(float* prev_output);
    void backward(float* prev_d_output);

    void forward_reset();
    void backward_reset();
};

__global__ void cudaSigmoidForward(float* prev_output, float* output, int N);

__global__ void backward_gradient_prev(float* prev_d_output, float* d_output, float* output,int N);