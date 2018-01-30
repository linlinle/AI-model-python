# -*- coding: utf-8 -*-
'''Prior to TensorFlow 1.2,,users had two options for feeding data to the TensorFlow training and eval pipelines:
        1.Feed data directly via feed_dict at each training session.run call.
        2.Use the queueing mechanisms in tf.train (e.g. tf.train.batch) and tf.contrib.train.
        3.Use helpers from a higher level framework like tf.contrib.learn or tf.contrib.slim .

    tarting in TensorFlow 1.2, there is a new system available for reading data into TensorFlow models: dataset iterators,
as found in the tf.contrib.data module.A dataset can be created from a batch data Tensor, a filename,
or a Tensor containing multiple filenames.
'''

import tensorflow as tf

# Training dataset consists of multiple files.
train_dataset = tf.contrib.data.TextLineDataset(train_files)

# Evaluation dataset uses a single file, but we may
# point to a different file for each evaluation round.
eval_file = tf.placeholder(tf.string, shape=())
eval_dataset = tf.contrib.data.TextLineDataset(eval_file)

# For inference, feed input data to the dataset directly via feed_dict.
infer_batch = tf.placeholder(tf.string, shape=(num_infer_examples,))
infer_dataset = tf.contrib.data.Dataset.from_tensor_slices(infer_batch)