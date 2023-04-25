import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# normal destribution
normal=tf.random.normal(shape=(4,4),mean=0,stddev=1)
print("normal destribution")
print(normal)


# uniform destribution
uniform=tf.random.uniform(shape=(1,5),minval=0,maxval=2)
print("uniform destribution")
print(uniform)

#ranege function
range=tf.range(start=5,limit=20,delta=3)# 5,8,11,14,17 3 by 3
print("range function")
print(range)

#casting 
y=tf.cast(range,tf.float32)
print(y)


