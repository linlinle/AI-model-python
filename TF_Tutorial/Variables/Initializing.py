# -*- coding: utf-8 -*-
import tensorflow as tf

my_variable = tf.get_variable("my_variable", [1, 2, 3])#dtype=tf.float32;initial=tf.glorot_uniform_initializer

with tf.Session() as session:
  # initializing all variables in the tf.GraphKeys.GLOBAL_VARIABLES collection.
  session.run(tf.global_variables_initializer())
  #initialize variables yourself
  session.run(my_variable.initializer)
  #prints the names of all variables which have not yet been initialized
  print(session.run(tf.report_uninitialized_variables()))


  # if the initial value of a variable depends on another variable's value,global_variables_initializer get an error
  # it is best to use variable.initialized_value() instead of variable
  v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
  w = tf.get_variable("w", initializer=v.initialized_value() + 1)