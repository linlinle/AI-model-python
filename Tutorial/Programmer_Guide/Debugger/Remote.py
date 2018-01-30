# -*- coding: utf-8 -*-
'''Often, your model is running on a remote machine or a process that you don't have terminal access to.
    To perform model debugging in such cases, you can use the offline_analyzer binary of tfdbg (described below).
    It operates on dumped data directories. This can be done to both the lower-level Session API and the higher-level
     Estimator and Experiment APIs

    Use:
        you can load and inspect the data in the dump directory on the shared storage by using the offline_analyzer
        binary of tfdbg

         >>   python -m tensorflow.python.debug.cli.offline_analyzer \
                --dump_dir=/shared/storage/location/tfdbg_dumps_1

     '''
import tensorflow as tf


#   If you interact directly with the tf.Session API in python,  you can configure the RunOptions proto that you call
# your Session.run() method with, by using the method tfdbg.watch_graph. This will cause the intermediate tensors and
# runtime graphs to be dumped to a shared storage location of your choice when the Session.run() call occurs
# (at the cost of slower performance).
from tensorflow.python import debug as tf_debug
# ... Code where your session and graph are set up...
fetches,feeds = '',''
with tf.Session() as session:
    run_options = tf.RunOptions()
    tf_debug.watch_graph(
          run_options,
          session.graph,
          debug_urls=["file:///shared/storage/location/tfdbg_dumps_1"])
    # Be sure to specify different directories for different run() calls.
    session.run(fetches, feed_dict=feeds, options=run_options)




#   The Session wrapper DumpingDebugWrapperSession offers an easier and more flexible way to generate file-system dumps
#  that can be analyzed offline. To use it, simply wrap your session in a tf_debug.DumpingDebugWrapperSession.
# Let your BUILD target depend on "//tensorflow/python/debug:debug_py
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug
my_watch_fn = ''
with tf.Session() as sess:
    sess = tf_debug.DumpingDebugWrapperSession(
        sess, "/shared/storage/location/tfdbg_dumps_1/", watch_fn=my_watch_fn)



#   Debugging Remotely-Running tf-learn Estimators and Experiments
# Let your BUILD target depend on "//tensorflow/python/debug:debug_py
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug
hooks = [tf_debug.DumpingDebugHook("/shared/storage/location/tfdbg_dumps_1")]
# Then this hook can be used in the same way as the LocalCLIDebugHook examples described earlier in this document.
# As the training and/or evalution of Estimator or Experiment happens, tfdbg creates directories having the following
# name pattern: /shared/storage/location/tfdbg_dumps_1/run_<epoch_timestamp_microsec>_<uuid>. Each directory
# corresponds to a Session.run() call that underlies the fit() or evaluate() call. You can load these directories and
# inspect them in a command-line interface in an offline manner using the offline_analyzer offered by tfdbg.

#   Use:
#        >> python -m tensorflow.python.debug.cli.offline_analyzer \
#                --dump_dir="/shared/storage/location/tfdbg_dumps_1/run_<epoch_timestamp_microsec>_<uuid>"