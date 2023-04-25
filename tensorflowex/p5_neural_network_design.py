# referance:https://keras.io/examples/nlp/
# referance:https://www.tensorflow.org/tutorials/keras/classification
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

print("x train=>",x_train.shape,"y train",y_train.shape) #(x train=> (60000, 28, 28) y train (60000,))     6000=num of images, 28x28 means pixel size

# neural network flattening and normalization because small values better for training
x_train=x_train.reshape(-1,28*28).astype("float32")/255.0
x_test=x_test.reshape(-1,28*28).astype("float32")/255.0


# print(type(x_train))
# if data is not in numpy(<class 'numpy.ndarray'>) data type
#x_train=tf.convert_to_tensor(x_train,dtype='float32')




# create training model for dataset
#complexing model: dropout layers(more advance concepts)

# the easy model

def createModel():
    model=keras.Sequential()
    model.add(keras.Input(shape=28*28))
    #hidden layers
    model.add(layers.Dense(512, activation="relu"))
    # hidden layer
    model.add(layers.Dense(256,activation="relu"))
    # final layerS
    model.add(layers.Dense(10))
    return model



myModel = createModel()

print(myModel.summary())


#compile the model(configure the model for training)

myModel.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

myModel.fit(x_train, y_train, batch_size=32,epochs=10,verbose=2)
myModel.evaluate(x_test,y_test, batch_size=32,verbose=2)

