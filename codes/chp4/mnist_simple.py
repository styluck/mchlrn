# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:05:58 2024

@author: lich5
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 对数据进行预处理
x_train, x_test = x_train / 255.0, x_test / 255.0

# 创建模型
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)


'''利用keras.Sequential命令创建神经网络'''
model = tf.keras.Sequential()
'''在模型中添加一个输出为128的全连接层，利用relu函数作为激活函数'''
model.add(tf.keras.layers.Dense(units=128,input_dim=784,activation='relu'))
'''在模型中添加一个输出为10的全连接层，利用softmax函数作为激活函数'''
model.add(tf.keras.layers.Dense(units=10,input_dim=128, activation='softmax'))

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(100, activation='relu'),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])


'''利用compile命令对模型进行编译，采用adam算法作为优化器，损失函数使用
sparse_categorical_crossentropy'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

# 训练模型
for step in range(100):
    # 每次训练一个批次
    cost = model.train_on_batch(x_train,y_train)
    if step % 10 == 0:
        print('cost:',cost)
# model.fit(x_train, y_train, epochs=5)

'''利用evalate命令评估模型'''
model.evaluate(x_test, y_test, verbose=2)


# model.save('my_model.h5')

# # 加载模型
# model = tf.keras.models.load_model('my_model.h5')
idx = np.random.randint(1e4,size=9)
images = x_test[idx,:]
y_ = y_test[idx]
# 测试模型
def plot_mnist_3_3(images, y_, y=None):
    assert images.shape[0] == len(y_)
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape([28,28]), cmap='binary')
        if y is None:
            xlabel = 'True: {}'.format(y_[i])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(y_[i], y[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

'''利用predict命令，输入x_test生成测试样本的测试值'''
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis = 1)

plot_mnist_3_3(images, y_, y_pred[idx])

