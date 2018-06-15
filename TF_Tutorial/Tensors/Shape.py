# -*- coding: utf-8 -*-
'''The shape of a tensor is the number of elements in each dimension.
Rank	Shape	            Dimension number	Example
0	    []	                     0-D	        A 0-D tensor. A scalar.
1	    [D0]	                 1-D	        A 1-D tensor with shape [5].
2	    [D0, D1]	             2-D	        A 2-D tensor with shape [3, 4].
3	    [D0, D1, D2]	         3-D	        A 3-D tensor with shape [1, 4, 3].
n	    [D0, D1, ... Dn-1]	     n-D	        A tensor with shape [D0, D1, ... Dn-1].
'''
import tensorflow as tf

#   Getting a tf.Tensor object's shape
my_image = tf.zeros([10, 299, 299, 3])
#make a vector of zeros with the same size as the number of columns in a given matrix
zeros = tf.zeros(tf.shape(my_image)[:1])

#   Changing the shape of a tf.Tensor
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content intoa 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20 matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a 4x3x5 tensor
