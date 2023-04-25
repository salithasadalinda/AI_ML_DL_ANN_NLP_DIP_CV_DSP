import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


#elementwise addition 
tensor1=tf.constant([20,2,1])
tensor2=tf.constant([21,22,4])

print(tensor1)
print(tensor2)


# all are can be use like a+b a/b a*b a-b
print("tensor addition")
tensor3=tf.add(tensor1,tensor2)# same as (tensor3=tensor1+tensor2)
print(tensor3)

print("tenser subtract")
tensor4=tf.subtract(tensor2,tensor1)
print(tensor4)

print("tenser elementwise division")
tensor5=tf.divide(tensor2,tensor1)
print(tensor5)

print("elem wise multiplication")
tensor6=tf.multiply(tensor1 ,tensor2)
print(tensor6)

print("dot product")
import numpy as np
tensor7=tf.constant([[1,2],[3,4]])
tensor8=tf.constant([[5,6],[7,8]])

print("tenor7")
print(tensor7)

print("tensor8")
print(tensor8)

print("dot product of tenser7 and tenser8")
tensor9=tf.tensordot(tensor7,tensor8,axes=1)
print(tensor9)
################################################copied from chatGpt################################################
# # Define two arrays using NumPy
# array1 = np.array([1, 2, 3])
# array2 = np.array([4, 5, 6])

# # Convert the arrays to TensorFlow tensors
# tensor1 = tf.constant(array1)
# tensor2 = tf.constant(array2)

# # Compute the dot product using tf.tensordot()
# dot_product = tf.tensordot(tensor1, tensor2, axes=1)

# # Print the result
# print(dot_product)

# lean how can i make numpy arrays and input into tensorflow










