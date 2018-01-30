# -*- coding: utf-8 -*-
'''A TensorFlow program relying on a pre-made Estimator typically consists of the following four steps:'''
import tensorflow as tf
#   1.Write one or more dataset input_fn.
feature_dict = 'a dictionary in which the keys are feature names and the values are' \
               ' Tensors (or SparseTensors) containing the corresponding feature data'
label = 'a Tensor containing one or more labels'
def input_fn(dataset):
   ...  # manipulate dataset, extracting feature names and the label
   return feature_dict, label

#   2.Define the feature columns:
#   Each tf.feature_column identifies a feature name, its type, and any input pre-processing.
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
# The first two feature columns simply identify the feature's name and type.
crime_rate = tf.feature_column.numeric_column('crime_rate')
#The third feature column also specifies a lambda(该程序用于将来缩放原始数据：):
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn='lambda x: x - global_education_mean')

#   3. Instantiate the relevant pre-made Estimator.
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )

#   4. Call a training, evaluation, or inference method.
#all Estimators provide a train method
# my_training_set is the function created in Step 1
my_training_set = ''
estimator.train(input_fn=my_training_set, steps=2000)