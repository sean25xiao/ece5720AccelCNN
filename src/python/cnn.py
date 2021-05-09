# Tensorflow CPU 
# run with python cnn_cpu.py 

import tensorflow as tf 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import layers 
import time 



# if __name__ = "__main__":

(x_train,y_train),(x_test,y_test)=mnist.load_data()  #可以直接用API加载MNIST的数据
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))


model = Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(5,5),input_shape=(28,28,1)))
model.add(layers.MaxPooling2D( pool_size = (4,4)))
model.add(layers.Flatten())
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary() # print summary of model structure

tic = time.perf_counter() # timing
model.fit(x_train, y_train, epochs=5, batch_size = 100) 
toc = time.perf_counter()

val_loss,val_acc=model.evaluate(x_test,y_test) 

print("Time cost = {:.3f}".format(toc-tic))
print("Test loss = {}. accuracy = {}".format(val_loss, val_acc))


