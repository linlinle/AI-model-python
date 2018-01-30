# -*- coding: utf-8 -*-
'''To inspect a tf.Tensor's data type use the Tensor.dtype property.'''
import tensorflow as tf

# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)