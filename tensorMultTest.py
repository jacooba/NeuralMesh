import numpy as np

a = np.arange(12.).reshape(3,4)
b = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])

print(a)
print(b)

print("")

print(np.tensordot(a,b, axes=([1],[1])))
print(np.tensordot(a,b, axes=([0,1],[0,1])))


c = np.array([[[1,1,1,1],[2,2,2,2],[3,3,3,3]], [[1,1,1,1],[2,2,2,2],[3,3,3,3]]])
print(np.tensordot(a,c, axes=([0,1],[1,2])))

print("")
print(np.roll(a, 1, axis=1))

print("")
print(np.matmul(c, np.identity(4)))

print("\n")
import os

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()
# a = tf.constant(np.roll(np.identity(3), 1, axis=0))[1,:]
# print(sess.run(a))

# print("")
# print(zip(*[("img",3),("img2", 5)]))

# print("")
# print(sum([False, True, True]))


# print("")
# arr1 = np.array([[1,1,1],[2,2,2],[3,3,3]])
# arr2 = np.array([[[1,1,1,1],[2,2,2,2],[3,3,3,3]],[[1,1,1,1],[2,2,2,2],[3,3,3,3]]])
# print(arr1.shape)
# print(arr2.shape)
# print(np.matmul(arr1, arr2))

# print("")
# arr1 = tf.constant(np.array([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]]))
# arr2 = tf.constant(np.array([[[1.,1.,1.,1.],[2.,2.,2.,2.],[3.,3.,3.,3.]],[[1.,1.,1.,1.],[2.,2.,2.,2.],[3.,3.,3.,3.]]]))
# #arr3 = tf.constant(np.array([[1.,1.,1.,1.],[2.,2.,2.,2.],[3.,3.,3.,3.]]))
# # print(arr1.shape)
# # print(arr2.shape)
# #mult = tf.matmul(arr1, arr2) #DOESNR WOEK IN TF, only NUMPY
# mult = tf.tensordot(arr1, arr2, axes=([0],[1]))
# add = arr1 + 100
# print(sess.run(mult))

# print(sess.run(add))

# print(np.array((92,2)))
to_shift = tf.constant(np.array( [ [[1. ,2.], [3., 4.], [5.,6.]] for batch in range(2) ] ))

D_SHIFT_MATRIX = tf.constant(np.roll(np.identity(3), 1, axis=0), dtype=tf.float64) # D_SHIFT_MATRIX * [] => shift all rows down 1
U_SHIFT_MATRIX = tf.constant(np.roll(np.identity(3), -1, axis=0), dtype=tf.float64)
R_SHIFT_MATRIX = tf.constant(np.roll(np.identity(2), 1, axis=1), dtype=tf.float64) # [] * R_SHIFT_MATRIX => shift all cols right 1
L_SHIFT_MATRIX = tf.constant(np.roll(np.identity(2), -1, axis=1), dtype=tf.float64)

shifted_inc_D = tf.transpose(tf.tensordot(D_SHIFT_MATRIX, to_shift, axes=([1],[1])), [1, 0, 2])
shifted_inc_U = tf.transpose(tf.tensordot(U_SHIFT_MATRIX, to_shift, axes=([1],[1])), [1, 0, 2])
shifted_inc_R = tf.tensordot(to_shift, R_SHIFT_MATRIX, axes=([2],[0]))
shifted_inc_L = tf.tensordot(to_shift, L_SHIFT_MATRIX, axes=([2],[0]))

print("down:", sess.run(shifted_inc_D), "\n")
print("up:", sess.run(shifted_inc_U), "\n")
print("right:", sess.run(shifted_inc_R), "\n")
print("left:", sess.run(shifted_inc_L), "\n")





