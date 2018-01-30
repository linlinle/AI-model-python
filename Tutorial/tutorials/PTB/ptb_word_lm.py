"""Example / benchmark for building a PTB_LSTM LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.

The data required for this example is in the data/ dir of the
PTB_LSTM dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import numpy as np
import tensorflow as tf

import Tutorial.tutorials.PTB.reader as reader
import Tutorial.tutorials.PTB.util as util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", os.path.join('D:/tmp', 'tensorflow', 'PTB', 'data'),
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", os.path.join('D:/tmp', 'tensorflow', 'PTB', 'logs'),
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    #LSTM反向传播展开步数
    self.num_steps = num_steps = config.num_steps
    #每个epoch内需要多少轮训练的迭代
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB_LSTM model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training # 训练标记
    self._input = input_  # PIBInput类的实例
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size # LSTM 节点数
    vocab_size = config.vocab_size  # 词汇表大小

    #将embedding计算限定在cpu中进行
    with tf.device("/cpu:0"):
      # 初始化embedding矩阵，训练过程中不断更新embedding矩阵的值
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      # embedding_lookup查询单词对应的向量表达获得inputs
      #shape=batch_size, num_stemps, size
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    # 如果为训练状态，输入层添加dropout
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # 得到输出与state
    output, state = self._build_rnn_graph(inputs, config, is_training)


    softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())#  权重
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())#  偏置
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)# 网络最后输出:softmax
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)# 汇总batch的误差
    self._final_state = state#  保留最终状态

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)# 定义学习速率变量_lr,并设为不可训练
    tvars = tf.trainable_variables()# 获取全部可训练参数

    # clip_by_global_norm设置梯度的最大范数max_grad_norm(Gradient Clipping)，可以防止梯度爆炸问题
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),#针对前面的_cost，计算tvars的梯度
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)# 定义优化器为GradientDescent优化器

    # 创建训练操作_train_op，apply_gradients将Clipping过的梯度应用到所有可训练参数tvars，
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())#  然后使用get_or_create_global_step生成全局同一的训练步数

    # 设置名为_new_lr的placeholder用以控制学习速率
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    # assign进行赋值
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training):
    """Build the inference graph using CUDNN cell."""
    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=config.num_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):
    '''设置默认LSTM单元
      一个cell，是n个hidden units
    '''
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell
    #Stack MultiCell
    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    # 定义好的cell会依次接收num_steps个输入然后产生最后的state（n-tuple，n表示堆叠的层数）
    # 但是一个batch内有batch_size这样的seq，因此就需要[batch_size，s]来存储整个batch每个seq的状态。
    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state

    # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()
        # 传入input和state到堆叠LSTM单元。inputs = [batch中第几个样本，样本中第几个单词，单词的向量表达的维度]
        (cell_output, state) = cell(inputs[:, time_step, :], state)#state是整个seq输入完之后得到的每层的state
                                                                  #states的shape=[batch_size, n(LSTMStateTuple)]
        outputs.append(cell_output)# 添加到输出列表 shape=[batch_size, num_steps, size]
    # 将output内容用concat串联到一起，并使用reshape转换为一维向量
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    '''外部控制学习速率'''
    #将学习速率值传入_new_lr这个placeholder，并执行_lr_update操作进行修改
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  #@property可以将返回变量设为只读，防止修改变量引发问题
  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name


class SmallConfig(object):
  """Small config.小模型参数设置"""
  init_scale = 0.1 #  网络中权重值的初始规模
  learning_rate = 1.0# 学习率初始值
  max_grad_norm = 5#  梯度的最大范数
  num_layers = 2 #LSTM堆叠层数
  num_steps = 20 #LSTM梯度反向传播的展开步数
  hidden_size = 200 #LSTM的隐含节点数
  max_epoch = 4 #初始学习率可以训练的epoch数，之后需要调整学习速率
  max_max_epoch = 13 #  总共可以训练的epoch数
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20 # 每个batch中样本数量
  vocab_size = 10000
  rnn_mode = BLOCK


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05#  希望权重初始值不要太大，有利于温和训练
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35  #
  hidden_size = 650 #
  max_epoch = 6 #
  max_max_epoch = 39  #
  keep_prob = 0.5 #
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


class TestConfig(object):
  """Tiny config, for testing."""
  # 只是为了测试使用，参数尽量使用最小值
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  rnn_mode = BLOCK


def run_epoch(session, model, eval_op=None, verbose=False):
  """定义训练一个epoch数据的函数"""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  #创建输出结果的字典表
  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }#如果有测评操作eval_op，一并加入
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  #进入训练循环，次数=epoch_size
  for step in range(model.input.epoch_size):
    feed_dict = {}
    #state是因为每训练一次batch会得到一个final_state,在进入下个batch的训练时，需要将final_state赋值给init_state
    for i, (c, h) in enumerate(model.initial_state):
      #将全部LSTM单元的state加入feed_dict中，
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)# 传入feed_dict并执行fetches，对网络进行一次训练
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps#累加num_steps到

    # 每完成10%的epoch，就进行一次结果显示
    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)))
  #平均cost的自然常数指数，是语言模型中用来比较模型性能的重要指标，越低代表模型输出的概率分布在预测样本上越好
  return np.exp(costs / iters)


def get_config():
  """Get model config."""
  config = None
  if FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB_LSTM data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

  #每个位置的元素表示这个word在vocabulary中的index
  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  #训练和测试配置要一样
  config = get_config()
  eval_config = get_config()

  eval_config.batch_size = 1
  eval_config.num_steps = 1
  #创建
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)#参数范围[-init_scale,init_scale]

    #利用PTBInput和PTBModel建立用来训练模型m，验证模型mvalid，测试模型mtest。
    #其中训练验证模型直接使用前面config，测试模型使用eval_config
    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()

    # 创建训练管理器sv
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
      for i in range(config.max_max_epoch):
        # 每个epoch循环内，先计算学习率衰减
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)#  更新lr
        # 执行一个epoch的训练，输出当前的学习率，训练和验证集的perplexity
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      #完成全部训练后，计算并输出模型在测试集上的perplexity
      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
