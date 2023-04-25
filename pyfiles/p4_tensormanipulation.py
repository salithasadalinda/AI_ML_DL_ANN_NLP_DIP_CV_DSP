import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


tensor=tf.constant([0,1,2,3,4,5,6,7,8,])

print("printing tensor")
print(tensor)

print("tensor[:]")
print(tensor[:])

# tensor slice
print("tensor[1:]", "tensor slice")
print(tensor[1:])
print("tensor[2:4]")
print(tensor[2:4])


# values skipping

print("tensor[::2]")
print(tensor[::2])   # skip from 2x2 values

#elements reverse ordering
print("reverse  ordering")
print(tensor[::-1])

#print index wise
print("indices")

indicesIndexes = tf.constant([0,3])
print(indicesIndexes)
indexedTensor = tf.gather(tensor, indices=indicesIndexes)
print(indexedTensor)

######################################
mat4x2=tf.constant([[10,20],[30,40],[50,60],[70,80]])
print("printing tensor4x2")
print(mat4x2)
print("get the first raw with indexing")
print(mat4x2[0])
print("get element from tensor")
print("first element=",mat4x2[0,:1],"  second element=",mat4x2[1,:1])
print("first two raws")
print(mat4x2[0:3,:1])


# reshape the tensor
tensor_reshape=tf.range(16)# until 16 the values were filled inside a tensor
print("reshaping tensor")
print(tensor_reshape)

tensor_reshaped=tf.reshape(tensor_reshape,(4,4))
print(tensor_reshaped)






