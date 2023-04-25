import tensorflow as tf
# referance:https://keras.io/examples/nlp/
# referance:https://www.tensorflow.org/tutorials/keras/classification
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

# basic example tensor creation for single scaler value
tensorInt_con = tf.constant(8,shape=(1,1))#for integer single value
print(tensorInt_con.shape)
print(tensorInt_con)

#basic example tensor creation for floot
tenserFloat_con = tf.constant(8.0,shape=(1,1))
print(tenserFloat_con.shape)
print(tenserFloat_con)

# make a matrix for value set
tenserInt_mat=tf.constant([[10,20,30,40,50],[1,2,3,4,5]],shape=(5,2),dtype=tf.float32)
print(tenserInt_mat.shape)
print(tenserInt_mat)

#matrix(unit matrix)
unit_matrix=tf.ones([3,4],dtype=tf.float32)
print(unit_matrix)

#matrix(zero matrix)
zero_matrix=tf.zeros([3,4],dtype=tf.float32)
print(zero_matrix)

#matrix(identity matrix)
eye_matrix=tf.eye(5)
print(eye_matrix)





