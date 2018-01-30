# -*- coding: utf-8 -*-
import tensorflow as tf


#   allow_growth : 根据需求分配
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


#   per_process_gpu_memory_fraction：定量分配
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)