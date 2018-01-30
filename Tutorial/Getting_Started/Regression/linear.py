# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''线性回归问题
        Y = X*W+b
'''

import tensorflow as tf
import os

#   初始化变量和模型参数，定义训练闭环中的运算
W = tf.Variable(tf.zeros([2,1]),name='weights')
b = tf.Variable(0.,name='bias')

def inference(x):
    # 计算推断模型在数据X上的输出，并将结果返回
    return tf.matmul(x,W)+b

def loss(x,y):
    # 依据训练数据X和期望输出Y计算损失
    y_predicted = inference(x)
    return tf.reduce_sum(tf.squared_difference(y,y_predicted))

def inputs():
    # 读取或生成训练数据X及其期望输出Y
    weight_age = [[84,86],[73,20],[65,52],[70,30],[76,57],[69,25],[63,28],[72,36],
                  [79,57],[75,44],[27,24],[89,31],[65,52],[57,23],[56,60],[69,48],
                  [60,34],[79,51],[75,50],[82,34],[59,46],[67,23],[85,37],[55,40],[63,30]]
    blood_fat_content = [354,190,405,263,451,302,288,385,402,365,209,290,346,254,395,
                         434,220,374,308,220,311,181,274,303,244]
    return tf.to_float(weight_age),tf.to_float(blood_fat_content)
def train(total_loss):
    # 依据计算的总损失训练或调整模型参数
    lr_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(lr_rate).minimize(total_loss)
def evaluate(sess,x,y):
    # 对训练得到的模型进行评估
    print('evaluate',sess.run(inference([[80.,25.]])))  # 303
    print('evaluate',sess.run(inference([[65.,25.]])))  # 256
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
    training_step= 10000
    for step in range(training_step):
        sess.run(train_op)
        # 处于调试目的，查看损失在训练过程中递减的情况
        if step%10 ==0:
            print('loss: ',sess.run([total_loss]))
        if step%(training_step-1) ==0:
            saver.save(sess,'tmp/',global_step=step)

    saver.save(sess,'tmp/',global_step=training_step)
    evaluate(sess,X,Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()


# with tf.Session() as sess:
#     # 模型设置
#     initial_step = 0
#     tf.global_variables_initializer().run()
#     X,Y = inputs()
#
#     total_loss = loss(X,Y)
#     train_op = train(total_loss)
#     training_step = 1000
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#
#     # 验证之前是否已经保存了检查点文件
#     ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
#     if ckpt and ckpt.model_checkpoint_path:
#         # 从检查点回复模型参数
#         saver.restore(sess,ckpt.model_checkpoint_path)
#         initial_step = int(ckpt.model_checkpoint_path.rspilt('-',1)[1])
#     for step in range(initial_step,training_step):...
#
#     sess.close()



