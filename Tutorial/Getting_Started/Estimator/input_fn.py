# -*- coding: utf-8 -*-
'''The input_fn is used to pass feature and target data to the train, evaluate, and predict methods of the Estimator.
 The user can do feature engineering or pre-processing inside the input_fn.'''
import tensorflow as tf
import numpy as np
import pandas as pd
import functools
######################################################################################
#The following code illustrates the basic skeleton for an input function:
def my_input_fn():
    feature_cols = 'A dict containing key/value pairs that map feature column names to ' \
                   'Tensors (or SparseTensors) containing the corresponding feature data.'
    labels = 'A Tensor containing your label (target) values: the values your model aims to predict.'

    # Preprocess your data here...

    # ...then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return feature_cols, labels
#######################################################################################
#Converting Feature Data to Tensors

# numpy input_fn.
x_data,y_data = 'numpy arrays','numpy arrays'
my_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_data)},
    y=np.array(y_data), )
# pandas input_fn.
x_data,y_data = ' pandas dataframes',' pandas dataframes'
my_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({"x": x_data}),
    y=pd.Series(y_data),)
#For sparse, categorical data (data where the majority of values are 0)
sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],
                                values=[6, 0.5],
                                dense_shape=[3, 5])#[[0, 6, 0, 0, 0]
                                                    # [0, 0, 0, 0, 0]
                                                    #[0, 0, 0, 0, 0.5]]
#######################################################################################
#Passing input_fn Data to Your Model
classifier = tf.estimator.DNNClassifier()
classifier.train(input_fn=my_input_fn, steps=2000)
#input_fn参数必须接收函数对象my_input_fn，而不是函数调用my_input_fn()
#TypeError:classifier.train(input_fn=my_input_fn(training_set), steps=2000)

#参数化input_fn.无需定义多个不同数据集的input_fn
# 1)
training_set = ''
def my_input_fn(data_set):
  ...
def my_input_fn_training_set():
  return my_input_fn(training_set)

classifier.train(input_fn=my_input_fn_training_set, steps=2000)
# 2)
classifier.train(
    input_fn=functools.partial(my_input_fn, data_set=training_set),
    steps=2000)
# 3)
classifier.train(input_fn=lambda: my_input_fn(training_set), steps=2000)
# 4）tf.estimator.inputs  create input_fn for numpy or pandas data sets.can control  more arguments
def get_input_fn_from_pandas(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame(...),
      y=pd.Series(...),
      num_epochs=num_epochs,
      shuffle=shuffle)

def get_input_fn_from_numpy(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.numpy_input_fn(
      x={...},
      y=np.array(...),
      num_epochs=num_epochs,
      shuffle=shuffle)