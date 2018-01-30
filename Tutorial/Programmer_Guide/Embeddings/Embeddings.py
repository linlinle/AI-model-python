# -*- coding: utf-8 -*-
import tensorflow as tf

vocabulary_size = 5
embedding_size = 100
word_ids = [1,2,3,4,5]
word_embeddings = tf.get_variable('word_embeddings',
    [vocabulary_size, embedding_size])
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
