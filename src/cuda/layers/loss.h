/*
 * Loss Layer
 */

class Loss
{
public:
    Loss(){};
    ~Loss(){};
    void static softmax(float* loss, float* logit, int label,int numClass);
};

__global__ void cudaSoftmax(float* loss, float* logit, int label,int numClass);
