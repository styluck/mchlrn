# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:16:01 2024

@author: lich5
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data(fname):
    """
    载入数据。
    """
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)


def eval_acc(label, pred):
    """
    计算准确率。
    """
    return np.sum(label == pred) / len(pred)


class SVM():
    """
    SVM模型。
    """

    def __init__(self, dim):
        
        self.W = tf.Variable(shape=[dim, 1], dtype=tf.float32, 
            initial_value=tf.random.uniform(shape=[dim, 1], minval=-0.1, maxval=0.1))
        
        self.b = tf.Variable(shape=[1], dtype=tf.float32, initial_value=tf.zeros(shape=[1]))
        
        self.trainable_variables = [self.W, self.b]


    @tf.function
    def get_xw(self, inp):
        inp = tf.cast(inp, tf.float32)
        pred = tf.matmul(inp, self.W) + self.b # shape(N, 1)
        # pred = tf.nn.sigmoid(pred)
        return pred

    @tf.function
    def predict(self, inp):
        """
        决策函数。
        """        
        pred = tf.sign(self.get_xw(inp))
        
        return pred
    
    @tf.function
    def compute_loss(self, pred, label):
        # if not isinstance(label, tf.Tensor):
        label = tf.cast(label, tf.float32)
            
        pred = tf.squeeze(pred, axis=1) 
        
        loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - pred*label))
        
        pred = tf.where(pred>0.5, tf.ones_like(pred), tf.zeros_like(pred))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(label, pred), dtype=tf.float32))
        return loss, accuracy
        
    @tf.function
    def train(self, optimizer, x, y):
        """
        训练模型。
        """
        with tf.GradientTape() as tape:
            pred = self.get_xw(x)
            loss, accuracy = self.compute_loss(pred, y)
            
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, accuracy


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)
    # 使用训练集训练SVM模型

    # 使用SVM模型预测标签
    x_train = data_train[:, :2]  # feature [x1, x2]
    y_train = data_train[:, 2]  # 真实标签
    
    
    dim = np.shape(x_train)[1]
    model = SVM(dim)  # 
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    for i in range(1000):
        loss, accuracy = model.train(opt, x_train, y_train)  # 训练模型
        if i%20 == 0:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')
    
    y_train_pred = model.predict(x_train)  # 预测标签
    x_test = data_test[:, :2]
    y_test = data_test[:, 2]
    y_test_pred = model.predict(x_test)

    # 评估结果，计算准确率
    acc_train = eval_acc(y_train, y_train_pred.numpy().squeeze())
    acc_test = eval_acc(y_test, y_test_pred.numpy().squeeze())
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))


    # 学习结果可视化
    C1 = np.where(y_test==1)[0]
    C2 = np.where(y_test==-1)[0]
    x1_min, x1_max = np.min(x_test[:,0]), np.max(x_test[:,0])
    x2_min, x2_max = np.min(x_test[:,1]), np.max(x_test[:,1])
    
    a,b = model.W.numpy()
    c = model.b.numpy()
    
    xx = np.arange(x1_max, step=0.1)    
    yy = a/-b * xx +c/-b
        
    f, ax = plt.subplots(figsize=(6,4))
    f.suptitle('Linear SVM Example', fontsize=15)
    plt.ylabel('Y')
    plt.xlabel('X')
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    
    plt.scatter(x_test[C1, 0], x_test[C1, 1], c='b', marker='+')
    plt.scatter(x_test[C2, 0], x_test[C2, 1], c='g', marker='o')
    plt.plot(xx,yy, label='fit_line')
    
# [EOF]