# -*- coding: utf-8 -*-
'''The rank of a tf.Tensor object is its number of dimensions.
Rank	Math entity
0	    Scalar (magnitude only)
1	    Vector (magnitude and direction)
2	    Matrix (table of numbers)
3	    3-Tensor (cube of numbers)
n	    n-Tensor (you get the idea)
'''
import tensorflow as tf
#Rank 0
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable((12.3, -4.85), tf.complex64)
#Rank 1
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([(12.3, -4.85), (7.5, -6.23)], tf.complex64)
#ranks 2
mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)
#Higher ranks
my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color


#Getting a tf.Tensor object's rank

r = tf.rank(my_image)
print(r)
# After the graph runs, r will hold the value 4.