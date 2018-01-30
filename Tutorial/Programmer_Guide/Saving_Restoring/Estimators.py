# -*- coding: utf-8 -*-

import tensorflow as tf

#Preparing serving inputs
feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  #将占位符添加到graph满足inference请求。
  #添加所需的其他操作，将输入数据转换为模型预期要求的特征张量。
  default_batch_size = 1
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')

  receiver_tensors = {'examples': serialized_tf_example}
  #传递feature_spec给tf.parse_example告诉解析器想要什么样的特性名称以及如何将它们映射到Tensors。
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

#Performing the expor 导出已训练的估算器
export_dir_base,estimator = '',tf.estimator.DNNClassifier()
estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn)