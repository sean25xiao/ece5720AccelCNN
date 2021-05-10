/*
 * Input Layer
 */

class Input
{
public:
    float *output;
    int O;
    Input(int width, int height);
    ~Input();
    void forward(float* data);
};
