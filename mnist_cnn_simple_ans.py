# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:03:44 2024

@author: lich5
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 对数据进行预处理
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.reshape(x_train, [-1, 28, 28, 1])
x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT") 
# y_train = tf.cast(y_train, tf.int64)
x_test = tf.reshape(x_test, [-1, 28, 28, 1])
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT") 
# y_test = tf.cast(y_test, tf.int64)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

'''减少计算时间，仅在60000个样本中取10000个进行学习'''
# train_ds = train_ds.shuffle(60000).batch(100)
# test_ds = test_ds.shuffle(10000).batch(10000)
train_ds = train_ds.take(10000).shuffle(20000).batch(100)
test_ds = test_ds.take(1000).shuffle(1000).batch(1000)


# 创建模型

model = tf.keras.Sequential()
# Convolutional Layer #1
# Has a default stride of 1
# Output: 28 * 28 * 6
model.add(tf.keras.layers.Conv2D(6,kernel_size=(5,5), activation='relu', padding='valid'))
# Pooling Layer #1
# Sampling half the output of previous layer
# Output: 14 * 14 * 6
model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))
# Convolutional Layer #2
# Output: 10 * 10 * 16
model.add(tf.keras.layers.Conv2D(16,(5,5), activation='relu', padding='valid'))
# Pooling Layer #2
# Output: 5 * 5 * 16
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# Reshaping output into a single dimention array for input to fully connected layer
model.add(tf.keras.layers.Flatten())
# Fully connected layer #1: Has 120 neurons
model.add(tf.keras.layers.Dense(units=120, activation='relu'))
# Fully connected layer #2: Has 84 neurons
model.add(tf.keras.layers.Dense(units=84, activation='relu'))
# Output layer, 10 neurons for each digit
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))


# 训练模型

start = time.perf_counter()
optimizer = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=7)
end = time.perf_counter() # time.process_time()
c=end-start 
print("程序运行总耗时:%0.4f"%c, 's') 

model.evaluate(test_ds)

# model.save('my_model.h5')
# model = tf.keras.models.load_model('my_model.h5')
idx = np.random.randint(1e4,size=9)
x = x_test.numpy().squeeze()
images = x[idx,:].squeeze()
y_ = y_test[idx]

# 测试模型
def plot_mnist_3_3(images, y_, y=None):
    assert images.shape[0] == len(y_)
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape([32,32]), cmap='binary')
        if y is None:
            xlabel = 'True: {}'.format(y_[i])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(y_[i], y[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis = 1)

plot_mnist_3_3(images, y_, y_pred[idx])

