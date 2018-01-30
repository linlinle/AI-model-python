# -*- coding: utf-8 -*-
'''TFDBG currently supports only training with tf-slim. To debug the training process,
    provide LocalCLIDebugWrapperSession to the session_wrapper argument of slim.learning.train()'''
import tensorflow as tf
from tensorflow.python import debug as tf_debug

train_op,logdir = '',''
# ... Code that creates the graph and the train_op ...
tf.contrib.slim.learning_train(
    train_op,
    logdir,
    number_of_steps=10,
    session_wrapper=tf_debug.LocalCLIDebugWrapperSession)