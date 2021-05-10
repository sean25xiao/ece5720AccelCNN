#include "layers/conv2d.h"
#include "layers/input.h"

class Model
{
private:
    // 这里创建各个Layer的实例
    double learning_rate;

    Input input_layer;
    Conv2d conv_1;
    Conv2d conv_2;

    void forward(double data[28][28]);
    void backward();

public:
    Model();
    ~Model();
    void feed(double data[28][28]);
    void test();
};
