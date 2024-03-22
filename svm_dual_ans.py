# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:38:28 2024

@author: lich5
"""
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

    def __init__(self, N, dim):
        self.N = N
        self.dim = dim
        self.lamb = tf.Variable(shape=[N], dtype=tf.float32, 
            initial_value=tf.ones(shape=[N]))
        
        self.trainable_variables = [self.lamb]
        
        self.W = tf.ones(dim)
        
        self.b = 0.
        
    # @tf.function
    def get_w(self, x, y):
        """
        计算 w
        """        
        tmp = (y*self.lamb)
        tmp = tf.repeat(tf.reshape(tmp,[self.N,1]),self.dim,axis=1)*x
        self.W = tf.reduce_sum(tmp, axis = 0)
    
    # @tf.function
    def get_xw(self, inp):
        """
        计算 y_pred = w*x + b
        """        
        inp = tf.cast(inp, tf.float32)
        pred = tf.matmul(inp, tf.reshape(self.W, [self.dim,1])) + self.b # shape(N, 1)
        # pred = tf.nn.sigmoid(pred)
        return pred

    # @tf.function
    def predict(self, inp):
        """
        预测标签。
        """        
        inp = tf.cast(inp, tf.float32)
        pred = tf.sign(tf.matmul(inp, tf.reshape(self.W, [self.dim,1]))+ self.b)
        
        return pred
    
    # @tf.function
    def compute_loss(self, x, y):
        # if not isinstance(label, tf.Tensor):
        self.get_w(x, y)
        loss = .5*tf.reduce_sum(self.W*self.W) - tf.reduce_sum(self.lamb) 
        + 1e3*tf.norm(self.lamb*y)**2
        
        # update self.b
        idx = tf.where(self.lamb>0)[0][0]
        y_ = y[idx]
        x_ = x[idx,:]
        # y_ =  tf.cast(y[idx], tf.float32)
        # x_ = tf.cast(x[idx,:], tf.float32)
        self.b = y_ + tf.reduce_sum(x_*self.W)
        
        return loss
        
    # @tf.function
    def train(self, optimizer, x, y):
        """
        训练模型。
        """
        with tf.GradientTape() as tape:
            
            loss = self.compute_loss(x, y)
            
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


if __name__ == '__main__':
    # 载入数据，实际实用时将x替换为具体名称
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)  # 数据格式[x1, x2, t]
    data_test = load_data(test_file)
    # 使用训练集训练SVM模型

    # 使用SVM模型预测标签
    x_train = tf.cast(data_train[:, :2] , tf.float32) # feature [x1, x2]
    y_train = tf.cast(data_train[:, 2] , tf.float32) # 真实标签
    
    
    N, dim = np.shape(x_train)
    model = SVM(N, dim)  # 
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    # opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    opt = tf.keras.optimizers.Adagrad(learning_rate=0.01)
    
    for i in range(1000):
        loss = model.train(opt, x_train, y_train)  # 训练模型
        if i%25 == 0:
            print(f'loss: {loss.numpy():.4}')
    
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