# -*- coding: utf-8 -*-

import math
import time
from datetime import datetime

import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

from ResNet_V2.resnet_v2 import resnet_v2_152
from resnet_utils import resnet_arg_scope

slim = tf.contrib.slim


def time_tensorflow_run(session, target,num_batches ,info_string):
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
  for i in xrange(num_batches + num_steps_burn_in):
    start_time = time.time()    #记录时间
    _ = session.run(target)     #每次迭代通过run执行
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:    #每10轮显示时间
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration

  mn = total_duration /num_batches   #  平均耗时
  vr = total_duration_squared /num_batches - mn * mn
  sd = math.sqrt(vr)    #   标准差
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, num_batches, mn, sd))


batch_size = 32
height,width = 224,224
inputs = tf.random_uniform((batch_size,height,width,3))
with slim.arg_scope(resnet_arg_scope()):
    net,end_points = resnet_v2_152(inputs,1000,is_training=False)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess,net,num_batches,"Forward")

