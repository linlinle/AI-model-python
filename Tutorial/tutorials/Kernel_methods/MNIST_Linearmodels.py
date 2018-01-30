# -*- coding: utf-8 -*-
import time

import numpy as np
import tensorflow as tf




def get_input_fn(dataset_split, batch_size, capacity=10000, min_after_dequeue=3000):

  def _input_fn():
    images_batch, labels_batch = tf.train.shuffle_batch(
        tensors=[dataset_split.images, dataset_split.labels.astype(np.int32)],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True,
        num_threads=4)
    features_map = {'images': images_batch}
    return features_map, labels_batch

  return _input_fn
data = tf.contrib.learn.datasets.mnist.load_mnist()
train_input_fn = get_input_fn(data.train, batch_size=256)
eval_input_fn = get_input_fn(data.validation, batch_size=5000)

#   Training a simple linear model

# Specify the feature(s) to be used by the estimator.
image_column = tf.contrib.layers.real_valued_column('images', dimension=784)
#Follow-The-Regularized-Leader (FTRL) optimization strategy
optimizer = tf.train.FtrlOptimizer(learning_rate=50.0, l2_regularization_strength=0.001)
kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(input_dim=784, output_dim=2000, stddev=5.0, name='rffm')
#a dictionary from feature_columns to a list of kernel mappings to be applied to the corresponding feature column.
kernel_mappers = {image_column: [kernel_mapper]}
estimator = tf.contrib.kernel_methods.KernelLinearClassifier(n_classes=10, optimizer=optimizer, kernel_mappers=kernel_mappers)

# Train.
start = time.time()
estimator.fit(input_fn=train_input_fn, steps=2000)
end = time.time()
print('Elapsed time: {} seconds'.format(end - start))

# Evaluate and report metrics.
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
print(eval_metrics)