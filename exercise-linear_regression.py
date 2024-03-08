import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as sk
from tensorflow.keras import optimizers, layers, Model
from tensorflow.keras.optimizers import SGD, Adam


def identity_basis(x,**kwargs):
    ret = np.expand_dims(x, axis=1)
    return ret

def multinomial_basis(x, feature_num=10):
    '''多项式基函数'''
#    x = np.expand_dims(x, axis=1) # shape(N, 1)
    
    #==========
    #todo '''请实现多项式基函数'''
    x = np.expand_dims(x, axis=1)
    x0 = np.ones_like(x)
    feat = [x0,x]
    for i in range(2,feature_num):
        feat.append(x**i)
    #==========
    ret = np.concatenate(feat, axis=1)
    return ret

def gaussian_basis(x, feature_num=10):
    '''高斯基函数'''
    #==========
    #todo '''请实现高斯基函数'''
    centers = np.linspace(0, 25, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x]*feature_num, axis=1)
    
    out = (x-centers)/width
    ret = np.exp(-0.5 * out ** 2)
    #==========
    
    return ret

def load_data(filename):
    """载入数据。"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)


class linearModel(Model):
    def __init__(self, ndim):
        super(linearModel, self).__init__()
        self.w = tf.Variable(
            shape=[ndim, 1], 
            initial_value=tf.random.uniform(
                [ndim,1], minval=-0.1, maxval=0.1, dtype=tf.float32))
        
    @tf.function
    def call(self, x):
        y = tf.squeeze(tf.matmul(x, self.w), axis=1)
        return y
    

# def main(x_train, y_train):
#     """
#     训练模型，并返回从x到y的映射。
    
#     """
#     #==========
#     #todo '''计算出一个优化后的w，请分别使用最小二乘法以及梯度下降两种办法优化w'''
#     #==========
    
#     def f(x):
#         phi0 = np.expand_dims(np.ones_like(x), axis=1)
#         phi1 = basis_func(x)
#         phi = np.concatenate([phi0, phi1], axis=1)
#         y = np.dot(phi, w)
#         return y

#     return f

def evaluate(ys, ys_pred):
    """评估模型。"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std

# 程序主入口（建议不要改动以下函数的接口）
if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.txt'
    # train_file = 'exercise-master\\chap2_linear_regression\\train.txt'
    # test_file = 'exercise-master\\chap2_linear_regression\\test.txt'
    
    
    # 载入数据
    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)
    
    # 构造基函数
    # basis_func = identity_basis
    # basis_func = gaussian_basis
    basis_func = multinomial_basis
    
    # phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    num_feature = 3
    phi_x = basis_func(x_train,feature_num=num_feature)
    
    dimx = phi_x.shape[1]
    # 使用线性回归训练模型，返回一个函数f()使得y = f(x)


    # 4.顺序模型squential的建立
    # 顺序模型是指网络是一层一层搭建的，前面一层的输出是后一层的输入
    model = tf.keras.Sequential()
    # 在模型中添加一个全连接层
    model.add(tf.keras.layers.Dense(units=10,input_dim=dimx,activation='tanh'))
    model.add(tf.keras.layers.Dense(units=1,input_dim=10))
    # sgd:Stochastic gradient descent，随机梯度下降法
    # mse:Mean Squared Error，均方误差
    model.compile(optimizer=Adam(0.01),loss='mse')
    
    # 4.查看模型的结构
    model.summary()
    # 训练3001个批次
    # history = model.fit(phi_x,y_train,epochs=1001)
    for step in range(1001):
        # 每次训练一个批次
        # cost = model.train_on_batch(x_train,y_train)
        cost = model.train_on_batch(phi_x,y_train)
        # 每500个batch打印一次cost值
        if step % 100 == 0:
            print('cost:',cost)
    
    y_train_pred = model.predict(phi_x)
    
    std = evaluate(y_train, y_train_pred)
    
    print('训练集预测值与真实值的标准差：{:.1f}'.format(std))
    
    # 计算预测的输出值
    phi_x_test = basis_func(x_test,feature_num=num_feature)
    
    y_test_pred = model.predict(phi_x_test)
    # 使用测试集评估模型
    std = evaluate(y_test, y_test_pred)
    # std1 = model.evaluate(phi_x_test,y_test)
    print('预测值与真实值的标准差：{:.1f}'.format(std))
    #显示结果
    plt.figure()
    plt.plot(x_train, y_train, 'ro', markersize=3)
    plt.plot(x_train, y_train_pred, 'ko')
    # plt.plot(x_test, y_test, 'ro', markersize=3)
    # plt.plot(x_test, y_test_pred, 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['test', 'pred'])
    plt.show()
    # plt.close('all')






