# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import math
import sys
import time

import tensorflow as tf
import vgg16

FLAGS = None





def time_tensorflow_run(session, target, feed, info_string):
  """Run the computation to obtain the target tensor and print timing stats.
    评估AlexNet每轮计算时间的函数。第一个输入Session，第二个输入需要评定的运算算子，第三个是测试的名称
  Args:
    session: the TensorFlow session to run the computation under.
    target: the target Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.

  Returns:
    None
  """
  num_steps_burn_in = 10    #   程序预热轮数，不计入计算
  total_duration = 0.0  #总时间
  total_duration_squared = 0.0  #平方和用以计算方差
  for i in range(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()    #记录时间
    _ = session.run(target,feed_dict = feed)     #每次迭代通过run执行
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:    #每10轮显示时间
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration

  mn = total_duration / FLAGS.num_batches   #  平均耗时
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)    #   标准差
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))



def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the Re_classifying to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    #   不使用ImageNet训练，只是用随机照片数据测试前馈和反馈计算的耗时
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,    #每轮迭代样本数
                                           image_size,
                                           image_size, 3],  #RGB
                                       dtype=tf.float32,
                                          stddev=1e-1))

    # Build a Graph that computes the logits predictions from the
    # inference Re_classifying.
    keep_prob = tf.placeholder(tf.float32)
    vgg = vgg16.Vgg16()
    vgg.build(images)

    # Build an initialization operation.
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Run the forward benchmark.
    time_tensorflow_run(sess, vgg.predictions,{keep_prob:0.8}, "Forward")     #统计前向传播运算时间

    # 反向传播时间计算
    # Add a simple objective so we can calculate the backward pass.
    objective = tf.nn.l2_loss(vgg.fc8)
    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, [[1,1],[2,2]])
    # Run the backward benchmark.
    time_tensorflow_run(sess, grad,{keep_prob:0.8}, "Forward-backward")


def main(_):
  run_benchmark()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Batch size.'
  )
  parser.add_argument(
      '--num_batches',
      type=int,
      default=100,
      help='Number of batches to run.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
