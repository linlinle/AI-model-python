# -*- coding: utf-8 -*-
'''This section of the guide describes the fundamentals of creating different kinds of Dataset and Iterator objects,
    and how to extract data from them.'''

import tensorflow as tf


#   Dataset structure
#construct a Dataset from some tensors
dataset1 = tf.contrib.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"
dataset2 = tf.contrib.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"
dataset3 = tf.contrib.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
#give names to each component of an element
dataset = tf.contrib.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
#transformations
dataset1 = dataset1.map(lambda x: ...)
dataset2 = dataset2.flat_map(lambda x, y: ...)




#   Creating an iterator
with tf.Session() as sess:
    #one_shot iterator :only supports iterating once through a dataset;not support parameterization
    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    for i in range(100):
      value = sess.run(next_element)
      assert i == value

    #initializable iterator:
    max_value = tf.placeholder(tf.int64, shape=[])
    dataset = tf.data.Dataset.range(max_value)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # Initialize an iterator over a dataset with 10 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 10})#要求显示的初始化
    for i in range(10):
        value = sess.run(next_element)
        assert i == value
    # Initialize the same iterator over a dataset with 100 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 100})
    for i in range(100):
        value = sess.run(next_element)
        assert i == value


    #reinitializable iterator：be initialized from multiple different Dataset；
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))#随机扰动
    validation_dataset = tf.data.Dataset.range(50)
    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                       training_dataset.output_shapes)
    next_element = iterator.get_next()
    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)
    # Run 20 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    for _ in range(20):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        for _ in range(100):
            sess.run(next_element)
        # Initialize an iterator over the validation dataset.
        sess.run(validation_init_op)
        for _ in range(50):
            sess.run(next_element)



#   Consuming values from an iterator
    dataset = tf.data.Dataset.range(5)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Typically `result` will be the output of a Re_classifying, or an optimizer's
    # training operation.
    result = tf.add(next_element, next_element)

    sess.run(iterator.initializer)
    print(sess.run(result))  # ==> "0"
    print(sess.run(result))  # ==> "2"
    print(sess.run(result))  # ==> "4"
    print(sess.run(result))  # ==> "6"
    print(sess.run(result))  # ==> "8"
    try:
        sess.run(result)
    except tf.errors.OutOfRangeError:
        print("End of dataset")  # ==> "End of dataset"
    #如果数据集的每个元素都具有嵌套结构，则返回值 Iterator.get_next()将是一个或多个tf.Tensor具有相同嵌套结构的对象：
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    iterator = dataset3.make_initializable_iterator()
    sess.run(iterator.initializer)
    next1, (next2, next3) = iterator.get_next()

