# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  in_units = 784 #隐含层输入节点数
  h1_units = 300 #隐含层输入节点数

  # Create the model
  x = tf.placeholder(tf.float32, [None, in_units])
  W1 = tf.Variable(tf.truncated_normal([in_units, h1_units]))#  初始化为截断的正态分布
  b1 = tf.Variable(tf.zeros([h1_units]))
  W2 = tf.Variable(tf.zeros([h1_units, 10]))
  b2 = tf.Variable(tf.zeros([10]))

  keep_prob = tf.placeholder(tf.float32)

  hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
  hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
  y = tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
  train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train一共采样3000个batch，每个batch100条样本，一共30万样本。相当于对全数据进行5轮(epoch)
  for _ in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob:0.8})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels,keep_prob:1.0}))
 # print(accuracy.eval({x:mnist.test.images,y_:mnist.test.images,keep_prob:1.0}))#测试部分keep_prob=1.0

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
