# -*- coding: utf-8 -*-

"""基于《Tensorflow 实战》的自编码器手写集实现"""

import numpy as np
import sklearn.preprocessing as prep #数据预处理
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init( fan_in, fan_out, constant = 1 ):
    '''权重初始化'''
    #标准的均匀分布的Xaiver初始化器，fan_in输入节点数量 fan_out输出结点数量
    low  = -constant * np.sqrt( 6.0 / ( fan_in + fan_out ) )
    high =  constant * np.sqrt( 6.0 / ( fan_in + fan_out ) )
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32 )

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale=0.1 ):
        self.n_input = n_input  #输入变量数
        self.n_hidden = n_hidden #隐含层节点数
        self.transfer = transfer_function   #隐含层激活函数，默认softplus
        self.scale = tf.placeholder( tf.float32 )#高斯噪声系数 ，默认0.1，占位符形式
        self.training_scale = scale
        network_weights = self._initialize_weights() #参数初始化
        self.weights = network_weights

        #定义网络结构
        self.x = tf.placeholder( tf.float32, [None, self.n_input] )#维度=n_input的placeholder
        #   建立提取特征的隐含层
        self.hidden = self.transfer( tf.add( tf.matmul(self.x + scale * tf.random_normal(( n_input, ) ),#输入x加入噪声
                                                       self.weights['w1'] ), self.weights['b1'] ))
        #数据复原，不需要transfer
        self.reconstruction = tf.add( tf.matmul( self.hidden, self.weights['w2'] ), self.weights['b2'] )

        #定义损失函数
        self.cost = 0.5 * tf.reduce_sum( tf.pow( tf.subtract( self.reconstruction, self.x ), 2 ) )
        self.optimizer = optimizer.minimize( self.cost )   #优化器，默认Adam

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run( init )#初始化全部参数

        print('begin to run session...')
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable( xavier_init( self.n_input, self.n_hidden ) )#需要Xavier初始化器
        all_weights['b1'] = tf.Variable( tf.zeros( [self.n_hidden], dtype = tf.float32 ) )
        all_weights['w2'] = tf.Variable( tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32) )
        all_weights['b2'] = tf.Variable( tf.zeros( [self.n_input], dtype = tf.float32 ) )
        return all_weights

    # 定义损失函数cost及执行一步训练
    def partial_fit(self, X):
        #session执行两个计算图的节点
        cost, opt = self.sess.run( (self.cost, self.optimizer), 
                                feed_dict = { self.x : X, self.scale : self.training_scale } )
        return cost
    #只求损失cost的函数
    def calc_total_cost( self, X ):
        #session执行一个计算图的节点，此函数在训练完毕后，在测试集上使用
        return self.sess.run( self.cost, feed_dict = { self.x : X, self.scale : self.training_scale } )

    def transform( self, X ):
        #返回字编码器隐含层的输出结果。目的提供一个接口来获取抽象后的特征，
        return self.sess.run( self.hidden, feed_dict = { self.x : X, self.scale : self.training_scale } )

    def generate( self, hidden = None ):
        #将隐含层输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为图像。
        if hidden == None:
            hidden = np.random.normal( size = self.weights['b1'] )
        return self.sess.run( self.reconstruction, feed_dict = { self.hidden : hidden } )

    def reconstruction( self, X ):
        #整体运行一边复原过程
        return self.sess.run( self.reconstruction, feed_dict = { self.x : X, self.scale : self.training_scale } )

    def getWeights( self ):
        #获取隐含层全重
        return self.sess.run( self.weights['w1'] )

    def getBiases( self ):
        return self.sess.run( self.weights['b1'] )


mnist = input_data.read_data_sets( 'D:/tmp/tensorflow/mnist/input_data', one_hot = True )

def standard_scale( X_train, X_test ):
    #对训练测试数据集进行标准化处理的函数
    preprocessor = prep.StandardScaler().fit( X_train )
    X_train = preprocessor.transform( X_train )
    X_test  = preprocessor.transform( X_test )
    return X_train, X_test

def get_random_block_from_data( data, batch_size ):
    #随机block的起始位置，然后顺序取到一个batch size自编码器实例。
    start_index = np.random.randint( 0, len(data) - batch_size )
    return data[ start_index : (start_index+batch_size)  ]

X_train, X_test  =standard_scale( mnist.train.images, mnist.test.images )

n_samples = int( mnist.train.num_examples )
training_epochs = 20
batch_size = 128
display_step = 1    #每隔一epoch就显示一次损失cost

#创建AGN
autoencoder = AdditiveGaussianNoiseAutoencoder( n_input = 784,n_hidden = 200,transfer_function = tf.nn.softplus,
                                                optimizer = tf.train.AdamOptimizer( learning_rate = 0.0001 ),
                                                scale = 0.01 )

for epoch in range( training_epochs ):
    avg_cost = 0
    total_batch = int( n_samples / batch_size )
    for i in range( total_batch ):
        batch_xs = get_random_block_from_data( X_train, batch_size )

        cost = autoencoder.partial_fit( batch_xs )
        avg_cost = cost / n_samples * batch_size


    if epoch % display_step == 0:
        print( "epoch : %04d, cost = %.9f" % ( epoch+1, avg_cost ) )

print( "Total cost : ",  str( autoencoder.calc_total_cost(X_test)))