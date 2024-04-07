import tensorflow as tf
import numpy as np

def softmax(x):
    ##########
    '''实现softmax函数，只要求对最后一维归一化，
    不允许用tf自带的softmax函数'''
    # prob_x = tf.nn.softmax(x)
    prob_x = tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=-1, keepdims=True)
    ##########
    return prob_x

test_data = np.random.normal(size=[10, 5])
np.sum(softmax(test_data).numpy() - tf.nn.softmax(test_data, axis=-1).numpy())**2 <0.0001

def sigmoid(x):
    ##########
    '''实现sigmoid函数， 不允许用tf自带的sigmoid函数'''
    # prob_x = tf.nn.sigmoid(x)
    prob_x = 1./(1. + tf.exp(-x))
    ##########
    return prob_x

test_data = np.random.normal(size=[10, 5])
np.sum(sigmoid(test_data).numpy() - tf.nn.sigmoid(test_data).numpy())**2 < 0.0001

def softmax_ce(label, test_data):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    pred = softmax(test_data)
    losses = -tf.reduce_sum(label*tf.math.log(pred), axis = 1)
    loss = tf.reduce_mean(losses)
    ##########
    return loss

test_data = np.random.normal(size=[10, 5])
pred = tf.nn.softmax(test_data)
label = np.zeros_like(test_data)
label[np.arange(10), np.random.randint(0, 5, size=10)]=1.

((tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, test_data))
  - softmax_ce(label,test_data))**2 < 0.0001).numpy()

def sigmoid_ce(label, test_data):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    prob = sigmoid(test_data)
    losses = -label*tf.math.log(prob) - (1 - label)*tf.math.log(1 - prob)
    loss = tf.reduce_mean(losses)
    ##########
    return loss

test_data = np.random.normal(size=[10])
label = np.random.randint(0, 2, 10).astype(test_data.dtype)
# print (label)

((tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label, test_data))- sigmoid_ce(label, test_data))**2 < 0.0001).numpy()




