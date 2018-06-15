# -*- coding: utf-8 -*-
'''逻辑回归问题
 来自 https://www.kaggle.com/c/titanic/data 的 Kaggle 竞赛数据集 Titanic

'''

import tensorflow as tf
import os

#   初始化变量和模型参数，定义训练闭环中的运算
W = tf.Variable(tf.zeros([5,1]),name='weights')
b = tf.Variable(0.,name='bias')

def combine_inputs(x):
    return tf.matmul(x,W)+b

def inference(x):
    # 计算推断模型在数据X上的输出，并将结果返回
    return tf.sigmoid(combine_inputs(x))

def loss(x,y):
    # 依据训练数据X和期望输出Y计算损失
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=combine_inputs(x),logits=y))

def read_csv(bath_size,file_name,record_default):
    filename_queue = tf.train.string_input_producer(
        [os.path.dirname(__file__)+"/"+file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key,value = reader.read(filename_queue)

    # decode_csv会将字符串(文本行)转换到具有指定默认值的由张量列构成的元组中
    # 它还会为每一列设置数据类型
    decode= tf.decode_csv(value,record_defaults=record_default)

    # 实际上会读取一个文件，并加载一个张量中的batch_size行
    return tf.train.shuffle_batch(decode,
                                  batch_size=bath_size,
                                  capacity=bath_size*50,
                                  min_after_dequeue=bath_size)

def inputs():
    # 读取或生成训练数据X及其期望输出Y
    passenger_id,survived,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked = \
    read_csv(100,'train.csv',[[0.0],[0.0],[0],[""],[""],[0.0],[0.0],[0.0],[""],[0.0],[""],[""]])
    # 转换属性数据(categorical data)
    is_first_class = tf.to_float(tf.equal(pclass,[1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))
    gender=tf.to_float(tf.equal(sex,["female"]))

    # 最终将所有特征排列在一个矩阵中，然后对该矩阵转置，使其每行对应一个样本，没列对应一种特征
    feature = tf.transpose(tf.stack([is_first_class,is_second_class,is_third_class
                                    ,gender,age]))
    survived = tf.reshape(survived,[100,1])
    return feature,survived

def train(total_loss):
    # 依据计算的总损失训练或调整模型参数
    lr_rate = 0.01
    return tf.train.GradientDescentOptimizer(lr_rate).minimize(total_loss)
def evaluate(sess,x,y):
    # 对训练得到的模型进行评估
    predicted = tf.cast(inference(x) > 0.5,tf.float32)
    print('evaluate',sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted,y),tf.float32))))



# 创建一个Saver对象
saver = tf.train.Saver()

#  在一个对话中启动数据流图，搭建流程
with tf.Session() as sess:
    # 模型设置
    tf.global_variables_initializer().run()
    X,Y = inputs()

    total_loss = loss(X,Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    # 实际的训练迭代次数
    training_step= 1000
    for step in range(training_step):
        sess.run(train_op)
        # 处于调试目的，查看损失在训练过程中递减的情况
        if step%100 ==0:
            print('loss: ',sess.run([total_loss]))
        if step%(training_step-1) ==0:
            saver.save(sess,'tmp/',global_step=step)

    saver.save(sess,'tmp/',global_step=training_step)
    evaluate(sess,X,Y)
    #evaluate(sess,[[0.,0.,1.,0.,39.]],[1])
    coord.request_stop()
    coord.join(threads)
    sess.close()



