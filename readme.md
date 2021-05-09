## Conda for python environment
install conda with 

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

sh Miniconda3-latest-Linux-x86_64.sh 
```

Configure your environment for tensorflow and tensorflow-gpu 

```shell
conda create -n tf-cpu python=3.7
conda install -n tf-cpu tensorflow

conda create -n tf-gpu python=3.7
conda install -n tf-cpu tensorflow-gpu

# you can check your conda current environment with `conda info -e`

conda activate tf-cpu # run this if you need to activate environment of tensorflow
conda activate tf-gpu # or run this if you need to activate environment of tensorflow-gpu
```
Once you configured both environments, you can activate them and then run your python script with `python yourfile.py`




## TODO

### 1. Batchsize affect tensorflow-gpu performance

Tensorflow with Gpu can accelerate CNN training, but this only came true when batch size is large enough. 

When batchsize = 1, tensorflow-gpu seems to be slower than tensorflow
but when batchsize = 100, 8s for tensorflow-gpu much faster than 160s for tensorflow.

As such, Training time vs Batch size plot is needed for both tensorflow and tensorflow-gpu






