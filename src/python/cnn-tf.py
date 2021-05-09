# Tensorflow (can run with tensorflow or tensorflow-gpu) 
# run with python cnn-tf.py 

import tensorflow as tf 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import layers 
import time 

# load data
(x_train,y_train),(x_test,y_test)=mnist.load_data()  # MNIST data can be loaded through this API
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))

# build model
model = Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(5,5),input_shape=(28,28,1)))
model.add(layers.MaxPooling2D( pool_size = (4,4)))
model.add(layers.Flatten())
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# print summary of model structure
model.summary() 

tic = time.perf_counter() # timing
model.fit(x_train, y_train, epochs=5, batch_size = 100) 
toc = time.perf_counter()

# evaluate this model with test data
val_loss,val_acc=model.evaluate(x_test,y_test) 

print("Time cost = {:.3f}".format(toc-tic)) # print the training time
print("Test loss = {}. accuracy = {}".format(val_loss, val_acc))