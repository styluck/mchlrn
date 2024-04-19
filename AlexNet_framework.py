# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:11:52 2024

@author: lich5
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

train_ds=tf.data.Dataset.from_tensor_slices((train_images,train_labels))
test_ds=tf.data.Dataset.from_tensor_slices((test_images,test_labels))

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(30,30))
# for i,(image,label) in enumerate(train_ds.shuffle(100000).take(20)):
#     #print(label)
#     ax=plt.subplot(5,5,i+1)
#     plt.imshow(image)
#     plt.title(CLASS_NAMES[label.numpy()[0]])
#     plt.axis('off')

def process_image(image,label):
    image=tf.image.per_image_standardization(image)
    image=tf.image.resize(image,(64,64))
    
    return image,label

'''
请补充模型
'''

# 训练模型
start = time.perf_counter()

history=model.fit(
    ?,
    epochs=1, #50
    validation_data=?
)


end = time.perf_counter() # time.process_time()
c=end-start 
print("程序运行总耗时:%0.4f"%c, 's') 


# 保存模型
model.save('cnn_model.h5')

# 加载模型
model = tf.keras.models.load_model('cnn_model.h5')

model.evaluate(?, verbose=2)

idx = np.random.randint(1e4,size=9)
images = test_images[idx,:]
y_ = test_labels[idx]

# 测试模型
def plot_cifar10_3_3(images, y_, y=None):
    assert images.shape[0] == len(y_)
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='binary')
        if y is None:
            xlabel = 'True: {}'.format(CLASS_NAMES[y_[i][0]])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(CLASS_NAMES[y_[i][0]], CLASS_NAMES[y[i]])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
'''利用predict命令，输入x_test生成测试样本的测试值'''
predictions = model.predict(?)
y_pred = np.argmax(predictions, axis = 1)

plot_cifar10_3_3(images, y_, y_pred)

f,ax=plt.subplots(2,1,figsize=(10,10)) 

#Assigning the first subplot to graph training loss and validation loss
ax[0].plot(model.history.history['loss'],color='b',label='Training Loss')
ax[0].plot(model.history.history['val_loss'],color='r',label='Validation Loss')

#Plotting the training accuracy and validation accuracy
ax[1].plot(model.history.history['accuracy'],color='b',label='Training  Accuracy')
ax[1].plot(model.history.history['val_accuracy'],color='r',label='Validation Accuracy')

plt.legend()