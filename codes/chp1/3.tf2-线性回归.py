import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD

x_data = np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)
y_data = x_data*0.1 + 0.2 + noise

# plt.scatter(x_data, y_data)
# plt.show()

# 构建一个顺序模型
model = tf.keras.Sequential()
# 在模型中添加一个全连接层
model.add(tf.keras.layers.Dense(units=1,input_dim=1))
# sgd:Stochastic gradient descent，随机梯度下降法
# mse:Mean Squared Error，均方误差
model.compile(optimizer=SGD(0.1),loss='mse')

# 训练3001个批次
for step in range(1001):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data,y_data)
    # 每500个batch打印一次cost值
    if step % 500 == 0:
        print('cost:',cost)

# x_data输入网络中，得到预测值y_pred
y_pred = model.predict(x_data)

# 显示随机点
plt.scatter(x_data,y_data)
# 显示预测结果
plt.plot(x_data,y_pred,'r-',lw=3)
plt.show()







