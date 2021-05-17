#include "layers/input.h"
#include "layers/conv2d.h"
#include "layers/softmax.h"
#include "layers/sigmoid.h"
#include "layers/common.h"

void fetchPrintMatrix(float *toprint, int channel, int width, int height)
{
    float tempOutput[channel][width][height];
    cudaMemcpy(tempOutput, toprint, sizeof(float) * width * height * channel, cudaMemcpyDeviceToHost);
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            printf("%.3f ", tempOutput[0][i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

bool functionalTest()
{
    nn::shape_3d input_shape = {28, 28, 1};
    Input input_layer = Input(input_shape);
    Conv2d conv_1 = Conv2d(input_layer.output_shape, 5, 1, 1, true);
    Conv2d conv_2 = Conv2d(conv_1.output_shape, 4, 1, 4, true);
    Softmax softmax_layer = Softmax(conv_2.output_shape);

    /* test func- setInputLayer */
    conv_1.setInputLayer(&input_layer);
    conv_2.setInputLayer(&conv_1);
    softmax_layer.setInputLayer(&conv_2);

    if (!conv_1.isPrevInput || conv_2.isPrevInput)
    {
        printf("Fail: isPrevInput\n");
        return false;
    }
    if (conv_2.prev_output != conv_1.output || conv_2.prev_d_output != conv_1.d_output)
    {
        printf("Fail: setInputLayer\n");
        return false;
    }

    /* test fea- forward */
    input_layer.clear();
    conv_1.clear();
    conv_2.clear();

    float sample[28][28];
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            sample[i][j] = 1.0;
        }
    }
    input_layer.feed(*sample);
    conv_1.forward();
    conv_2.forward();
    float conv2Output[1][6][6];
    cudaMemcpy(conv2Output, conv_2.output, sizeof(float) * 36, cudaMemcpyDeviceToHost);
    if (conv2Output[0][0][0] != 400 || conv2Output[0][5][5] != 400)
    {
        printf("Failed: forward [make sure weight=1 and bias=0]\n");
    }

    /* test fea- softmax */
    float sample2[9];
    for (int i = 0; i < 9; i++)
    {
        sample2[i] = 1;
    }
    nn::shape_3d input_shape2 = {3, 3, 1};
    Input input_layer2 = Input(input_shape2);
    Softmax softmax_layer2 = Softmax(input_layer2.output_shape);
    softmax_layer2.setInputLayer(&input_layer2);

    input_layer2.clear();
    softmax_layer2.clear();

    input_layer2.feed(sample2);
    softmax_layer2.feed(0);

    softmax_layer2.forward();
    softmax_layer2.backward();

    // fetchPrintMatrix(softmax_layer2.output,1,3,3);
    // printf("\n");
    // fetchPrintMatrix(softmax_layer2.prev_d_output,1,3,3);
    return true;
}

bool NetworkTest()
{
    // Data
    float sample[28][28];
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            sample[i][j] = nn::generateUniformRandom(1);
        }
    }

    int label = 0;

    // Build network
    nn::shape_3d input_shape = {28, 28, 1};
    Input input_layer = Input(input_shape);
    Conv2d conv_1 = Conv2d(input_layer.output_shape, 5,6,1);
    Conv2d conv_2 = Conv2d(conv_1.output_shape,4,6,4);
    Conv2d conv_3 = Conv2d(conv_2.output_shape,6,10,1);
    Sigmoid sigm_3 = Sigmoid(conv_3.output_shape);
    Softmax softmax_layer = Softmax(sigm_3.output_shape);

    conv_1.setInputLayer(&input_layer);
    conv_2.setInputLayer(&conv_1);
    conv_3.setInputLayer(&conv_2);
    sigm_3.setInputLayer(&conv_3);
    softmax_layer.setInputLayer(&sigm_3);

    float loss;    
    float learning_rate = 1e-3;

    for(int iter=0; iter<10000; iter++)
    {
        // clear
        input_layer.clear();
        conv_1.clear();
        conv_2.clear();
        conv_3.clear();
        sigm_3.clear();
        softmax_layer.clear();

        // feed
        input_layer.feed(*sample);
        softmax_layer.feed(label);

        // forward
        conv_1.forward();
        conv_2.forward();
        conv_3.forward();
        sigm_3.forward();
        softmax_layer.forward();

        // backward
        softmax_layer.backward();
        sigm_3.backward();
        conv_3.backward();
        conv_2.backward();
        conv_1.backward();

        // update
        nn::apply_grad(conv_1.weight, conv_1.d_weight, learning_rate*10, conv_1.weight_fdim);
        nn::apply_grad(conv_1.bias, conv_1.d_bias, learning_rate, conv_1.bias_fdim);
        
        nn::apply_grad(conv_2.weight, conv_2.d_weight, learning_rate*5, conv_2.weight_fdim);
        nn::apply_grad(conv_2.bias, conv_2.d_bias, learning_rate, conv_2.bias_fdim);

        nn::apply_grad(conv_3.weight, conv_3.d_weight, learning_rate*2, conv_3.weight_fdim);
        nn::apply_grad(conv_3.bias, conv_3.d_bias, learning_rate, conv_3.bias_fdim);
        
        // compute loss 
        if(iter % 100 == 0)
        {
            loss = softmax_layer.computeLoss();
            printf("Iter %d, loss=%.5f\n", iter, loss);
            if(iter % 1000 ==0)
            {
                fetchPrintMatrix(softmax_layer.output, 1, 1, 10);
                fetchPrintMatrix(softmax_layer.prev_d_output, 1, 1, 10);
            }
        }
    }
    
    return true;
}


int main(int argc, const char **argv)
{
    if (NetworkTest())
    {
        printf("You passed the function test!\n");
    }
    return 0;
}
