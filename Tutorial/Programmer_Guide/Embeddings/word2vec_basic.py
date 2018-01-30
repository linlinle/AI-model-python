"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download and Read the data.
url = 'http://mattmahoney.net/dc/'


# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  local_filename = os.path.join(os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/word2vec'), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')
  return local_filename

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()	#将数据转成单词的列表
  return data

def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  #Counter统计单词列表中单词的频数，most_common取top50000频数的单词作为vocabulary
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()	#创建一个dict，将top词汇的vocabulary放入其中
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:	#对每一个单词编号
    index = dictionary.get(word, 0)	#是否出现在dictionary中，
    if index == 0:  # dictionary['UNK']
      unk_count += 1	#统计数量
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  #返回转换后的编码(data)、每个单词的频数统计(count)、词汇表(dictionary)、及其反转形式(reverse_dictionary)
  return data, count, dictionary, reversed_dictionary

def generate_batch(batch_size, num_skips, skip_window):  # (batch大小,对每个单词生成多少样本,单词最远联系距离)
  global data_index
  assert batch_size % num_skips == 0  # 确保batch_size是num_skips的倍数
  assert num_skips <= 2 * skip_window  # 确保num_skips不大于两倍的skip_window
  # 将batch，labels初始化为数组
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [单词本身和前后的单词]
  buffer = collections.deque(maxlen=span)  # 最大容量为span的deque双向队列，

  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])  # 把span个单词顺序读入buffer作为初始值，后续数据替换掉前面数据
  data_index += span
  # 每次循环对一个目标函数生成样本
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    # 每次循环对一个语境单词(非目标单词)生成样本
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer[:] = data[:span]
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # 两次循环后我们获得batch_size个训练样本
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.解压下载的文件
vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the vocabulary词汇表 and replace rare words with UNK token.
vocabulary_size = 50000	#取top50000频数的单词
# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.删除原始单词列表
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0	#单词序号


# Step 3: Function to 生成训练样本 for the skip-gram model.
#将“the quick brown fox jumped over the lazy dog”转换为(quick,the)(quick,brown)(brown,quick)(brown,fox)等


#generate_batch测试
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.稠密向量的维度
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on.验证相关性数据
valid_window = 100  # Only pick dev samples in the head of the distribution.验证单词只从频数最高的100个中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

#定义网络结构
graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)#转为constant

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/gpu:0'):
    # Look up embeddings for inputs.随机生成所有单词的词向量embedding
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)	#查找输入train_inputs对应的向量embed

    # Construct the variables for the NCE loss初始化权重系数
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  #使用nce_loss计算学习出的词向量embedding在训练数据上的loss，并使用reduce_mean进行汇总
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))	#计算嵌入向量embedding的L2范数
  normalized_embeddings = embeddings / norm	#得到标准化后的embedding
  valid_embeddings = tf.nn.embedding_lookup(		#查询验证单词的嵌入向量
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(	#计算验证单词的嵌入向量与词汇表中所有单词的相似性
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()执行一次优化器运算和损失计算
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:#计算一次验证单词与全部单词的相似度，显示最相似8个单词
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Testing final embedding
input_dictionary = dict([(v, k) for (k, v) in reverse_dictionary.items()])

# 找寻france, paris和rome的word index
test_word_idx_a = input_dictionary.get('france')
test_word_idx_b = input_dictionary.get('paris')
test_word_idx_c = input_dictionary.get('rome')

# 在final_embeddings(也就是我们系统学到的embedding)里寻找上述词
a = final_embeddings[test_word_idx_a, :]
b = final_embeddings[test_word_idx_b, :]
c = final_embeddings[test_word_idx_c, :]

# 通过algebra的方式寻找预测词
ans = c + (a - b)
similarity = final_embeddings.dot(ans)

print(similarity.shape)
print(similarity[0:10])


# 选取最近的4个词展示出来
top_k = 4
nearest = (-similarity).argsort()[0:top_k + 1]
print(nearest)
for k in xrange(top_k + 1):
  close_word = reverse_dictionary[nearest[k]]
  print(close_word)

# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.


try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  #使用TSNE实现降维：128 ->  2
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500 #显示词频最高的500个
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/word2vec', 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
