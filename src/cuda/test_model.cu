#include "model.h"
#include <cstdio>

int main()
{
    float learning_rate = 0.001;
    Model cnn = Model(learning_rate);

    float data[28][28];
    for(int i=0; i<28; i++){
        for(int j=0; j<28; j++){
            data[i][j] = 10 * float(rand()) / float(RAND_MAX);            
        }
    }
    float err;

    for(int iter=1; iter < 20; iter++){
        err += cnn.feed(data, 3, true);
        printf("Iter %d: err = %f\n",iter, err);        
    }
    
    return 0;
}