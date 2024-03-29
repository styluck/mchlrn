# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:08:28 2024

@author: 6
"""

import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm
import numpy as np

class SoftmaxRegression():
    def __init__(self):
        '''============================='''
        #todo 填空一，构建模型所需的参数 self.W, self.b 可以参考logistic-regression-exercise
        self.W = tf.Variable(shape=[2, 5], dtype=tf.float32, 
            initial_value=tf.random.uniform(shape=[2, 5], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(shape=[5], dtype=tf.float32, initial_value=tf.zeros(shape=[5]))
        '''============================='''
        
        self.trainable_variables = [self.W, self.b]
        
    @tf.function
    def __call__(self, inp):
        inp = tf.cast(inp, tf.float32)
        pred = tf.matmul(inp, self.W) + self.b # shape(N, 3)
        # pred = tf.nn.softmax(pred)
        return pred    
    
@tf.function
def compute_loss(pred, label):
    # label = y
    label = tf.one_hot(tf.cast(label, dtype=tf.int32), 
                       dtype=tf.float32, depth=5)
    '''============================='''
    #输入label shape(N, 3), pred shape(N, 3)
    #输出 losses shape(N,) 每一个样本一个loss
    #todo 填空二，实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)
    
    # losses = -tf.reduce_sum(label*tf.math.log(pred), axis = 1)
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred)
    '''============================='''
    loss = tf.reduce_mean(losses)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label,axis=1), 
                                               tf.argmax(pred, axis=1)), 
                                      dtype=tf.float32))
    return loss, accuracy

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss, accuracy = compute_loss(pred, y)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy

if __name__=='__main__':
    dot_num = 100
    x_1 = np.random.normal(3., 1, dot_num)
    y_1 = np.random.normal(6., 1, dot_num)
    y = np.ones(dot_num)
    C1 = np.array([x_1, y_1, y]).T
    
    x_2 = np.random.normal(6., 1, dot_num)
    y_2 = np.random.normal(3., 1, dot_num)
    y = np.zeros(dot_num)
    C2 = np.array([x_2, y_2, y]).T
    
    x_3 = np.random.normal(7., 1, dot_num)
    y_3 = np.random.normal(7., 1, dot_num)
    y = np.ones(dot_num)*2
    C3 = np.array([x_3, y_3, y]).T
    
    x_4 = np.random.normal(1., 1, dot_num)
    y_4 = np.random.normal(1., 1, dot_num)
    y = np.ones(dot_num)*3
    C4 = np.array([x_4, y_4, y]).T
    
    x_5 = np.random.normal(9., 1, dot_num)
    y_5 = np.random.normal(9., 1, dot_num)
    y = np.ones(dot_num)*4
    C5 = np.array([x_5, y_5, y]).T
    
    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
    plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')
    plt.scatter(C4[:, 0], C4[:, 1], c='k', marker='^')
    plt.scatter(C5[:, 0], C5[:, 1], c='c', marker='.')
    
    data_set = np.concatenate((C1, C2, C3, C4, C5), axis=0)
    np.random.shuffle(data_set)
    
    epsilon = 1e-12
    model = SoftmaxRegression()
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    x1, x2, y = list(zip(*data_set))
    x = list(zip(x1, x2))
    for i in range(1000):
        loss, accuracy = train_one_step(model, opt, x, y)
        if i%50==49:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')
    
    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
    plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')
    plt.scatter(C4[:, 0], C4[:, 1], c='k', marker='^')
    plt.scatter(C5[:, 0], C5[:, 1], c='c', marker='.')
    
    x = np.arange(0., 10., 0.1)
    y = np.arange(0., 10., 0.1)
    
    X, Y = np.meshgrid(x, y)
    inp = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
    print(inp.shape)
    Z = model(inp)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(X.shape)
    plt.contour(X,Y,Z)
    plt.show()



