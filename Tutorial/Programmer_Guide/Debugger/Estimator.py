# -*- coding: utf-8 -*-

'''Currently, tfdbg can debug the fit() evaluate() methods of tf-learn Estimators. To debug Estimator.fit(),
    create a LocalCLIDebugHook and supply it in the monitors argument. '''
# First, let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug
import tensorflow as tf

# Create a LocalCLIDebugHook and use it as a monitor when calling fit().
hooks = [tf_debug.LocalCLIDebugHook()]
training_set,test_set = '',''
classifier = tf.estimator.Estimator()

#   To debug Estimator.fit()
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=1000,
               monitors=hooks)

#   To debug Estimator.evaluate()
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target,
                                     hooks=hooks)["accuracy"]