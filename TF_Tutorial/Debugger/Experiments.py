# -*- coding: utf-8 -*-
'''Experiment is a construct in tf.contrib.learn at a higher level than Estimator.
It provides a single interface for training and evaluating a Re_classifying.
To debug the train() and evaluate() calls to an Experiment object,
you can use the keyword arguments train_monitors and eval_hooks, respectively,
when calling its constructor.'''

# First, let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
import tensorflow as tf
from tensorflow.python import debug as tf_debug
FLAGS = None

hooks = [tf_debug.LocalCLIDebugHook()]
experiment = tf.contrib.learn.Experiment()
classifier = tf.contrib.learn.DNNClassifier()
iris_input_fn = ''


ex = experiment.Experiment(classifier,
                           train_input_fn=iris_input_fn,
                           eval_input_fn=iris_input_fn,
                           train_steps=FLAGS.train_steps,
                           eval_delay_secs=0,
                           eval_steps=1,
                           train_monitors=hooks,
                           eval_hooks=hooks)

ex.train()
accuracy_score = ex.evaluate()["accuracy"]