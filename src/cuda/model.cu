#include "model.h"

// 一些初始化的工作
Model::Model(){
    learning_rate = lr;
    // 可能需要初始化一些memory?

    // Layers的配置
    input_layer = Input(28,28);
    conv_1 = Conv2d(28,28,1,5,6,1);
    conv_2 = Conv2d(24,24,6,5,6,4);
};

// 模型训练
void Model::feed(double data[28][28]){
    forward(data);
    backward(data);
};

// 前向计算过程
void Model::forward(double data[28][28]){
    input_layer.forward(data);
    conv_1.forward(input.output);
    conv_2.forward(conv_1.output);
    
};

// 反向传播过程
void Model::backward(){
    conv_2.backward();
    conv_1.backward(conv_1.d_output);
};

// 打印测试结果
void Model::test(){

}
