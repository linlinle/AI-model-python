# -*- coding: utf-8 -*-

import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from CNN.Alexnet.Finetune.datagenerator import ImageDataGenerator

from CNN.Alexnet.Finetune.alexnet import AlexNet

# Path to the textfiles for the trainings and validation set
train_file = 'E:/baidu/datasets/preprocessing/train.txt'
val_file = 'E:/baidu/datasets/preprocessing/val.txt'
test_file = 'E:/baidu/datasets/preprocessing/test.txt'
train_layers = ['fc8']


# Learning params
learning_rate = 0.01
num_epochs = 20
batch_size = 10

# Network params
dropout_rate = 0.8
num_classes = 100


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, num_classes, train_layers)
# Link variable to model output
score = model.fc8

# Op for calculating the loss
with tf.name_scope("cross_ent"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))

with tf.name_scope("train"):
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.name_scope("predict"):
    predicted = tf.argmax(score, axis=1)



# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file,horizontal_flip = True, shuffle = True)
val_generator = ImageDataGenerator(val_file, shuffle = False)
test_generator = ImageDataGenerator(test_file, shuffle = False)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)
test_batches_per_epoch = np.floor(test_generator.data_size / batch_size).astype(np.int16)

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            sess.run(train_op, feed_dict={x: batch_xs,
                                          y: batch_ys,
                                          keep_prob: dropout_rate})
            step += 1

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: batch_ty,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()

    predicted_list = []
    for _ in range(test_batches_per_epoch):
        batch_tx, batch_ty = test_generator.next_batch(batch_size)
        predicte_batch = sess.run(predicted, feed_dict={x: batch_tx,
                                                        y: batch_ty,
                                                        keep_prob: 1.})
        predicted_list = np.hstack((predicted_list, predicte_batch))

a = pd.read_csv('E:/baidu/datasets/preprocessing/test.txt', delim_whitespace=True)
a['0'] = predicted_list.astype(np.int32)[1:]
a.rename(columns={'0': str(int(predicted_list[0]))}, inplace=True)
a.to_csv('E:/baidu/datasets/result.csv', index=False, sep=' ')