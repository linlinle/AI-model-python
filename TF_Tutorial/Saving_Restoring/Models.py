# -*- coding: utf-8 -*-
'''第二个参数tags是给当前需要保存的graph一个标签，标签名可以自定义，在之后载入模型的时候，
需要根据这个标签名去查找对应的MetaGraphDef，找不到就会报如
RuntimeError: MetaGraphDef associated with tags 'foo' could not be found in SavedModel
这样的错。标签也可以选用系统定义好的参数，如tf.saved_model.tag_constants.SERVING
与tf.saved_model.tag_constants.TRAINING。'''

import tensorflow as tf


#   Building a SavedModel
export_dir ='temp/Savemodel'
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.TRAINING],)
# Add a second MetaGraphDef for inference.
with tf.Session(graph=tf.Graph()) as sess:
  builder.add_meta_graph(['tag_string'])
builder.save()




#   Loading a SavedModel in Python
#The load operation requires the following information:
# *The session in which to restore the graph definition and variables.
# *The tags used to identify the MetaGraphDef to load.
# *The location (directory) of the SavedModel.
tf.reset_default_graph()
with tf.Session(graph=tf.Graph()) as sess:
   tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], export_dir)
   x = sess.graph.get_tensor_by_name('input_x:0')
   y = sess.graph.get_tensor_by_name('predict_y:0')
   # 实际的待inference的样本
   _x = ...
   sess.run(y, feed_dict={x: _x})