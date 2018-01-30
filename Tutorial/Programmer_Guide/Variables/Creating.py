# -*- coding: utf-8 -*-
import tensorflow as tf


my_variable = tf.get_variable("my_variable", [1, 2, 3])#dtype=tf.float32;initial=tf.glorot_uniform_initializer
#optionally specify the dtype and initializer to tf.get_variable.
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer)
#initialize a tf.Variable to have the value of a tf.Tensor
other_variable = tf.get_variable("other_variable", dtype=tf.int32,
  initializer=tf.constant([23, 42]))


#   Variable collections
#tf.GraphKeys.GLOBAL_VARIABLES --- variables that can be shared across multiple devices,
#tf.GraphKeys.TRAINABLE_VARIABLES--- variables for which TensorFlow will calculate gradients.
#don't want a variable to be trainable
my_local = tf.get_variable("my_local", shape=(),collections=[tf.GraphKeys.LOCAL_VARIABLES])#也可以按照下面方法
my_non_trainable = tf.get_variable("my_non_trainable", shape=(),trainable=False)
#use your own collections
tf.add_to_collection("my_collection_name", my_local)
print(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)[0])


#   Device placement
#place variables on particular devices
with tf.device("/device:GPU:0"):
  v = tf.get_variable("v", [1])