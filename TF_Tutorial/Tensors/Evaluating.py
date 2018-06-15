# -*- coding: utf-8 -*-
'''The eval method only works when a default tf.Session is active'''

import tensorflow as tf

p = tf.placeholder(tf.float32)

with tf.Session():
    constant = tf.constant([1, 2, 3])
    tensor = constant * constant
    print(tensor.eval())

    # value might depend on dynamic information that is not available
    t = p + 1.0
    #t.eval()   This will fail, since the placeholder did not get a value.
    print(t.eval(feed_dict={p: 2.0}))  # This will succeed because we're feeding a value
    # to the placeholder.