import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    #normalize
    x = x/255.0
    x_test = x_test/255.0
    
    return (x, y), (x_test, y_test)

class myModel:
    def __init__(self):
        ####################
        '''声明模型对应的参数'''
        ####################
    def __call__(self, x):
        ####################
        '''实现模型函数体，返回未归一化的logits'''
        ####################
        return logits
         
    @tf.function
    def compute_loss(self, logits, labels):
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels, logits))
        # return tf.reduce_mean(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         logits=logits, labels=labels))
    
    @tf.function
    def compute_accuracy(self, logits, labels):
        predictions = tf.argmax(logits, axis=1)
        return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
    
    @tf.function
    def train_one_step(self, optimizer, x, y):
        with tf.GradientTape() as tape:
            logits = self.__call__(x)
            loss = self.compute_loss(logits, y)
    
        # compute gradient
        trainable_vars = [self.W1, self.W2, self.b1, self.b2]
        grads = tape.gradient(loss, trainable_vars)
        for g, v in zip(grads, trainable_vars):
            v.assign_sub(0.01*g)
    
        accuracy = self.compute_accuracy(logits, y)
    
        # loss and accuracy is scalar tensor
        return loss, accuracy
    
    @tf.function
    def test(self, model, x, y):
        logits = self.__call__(x)
        loss = self.compute_loss(logits, y)
        accuracy = self.compute_accuracy(logits, y)
        return loss, accuracy

    def predict(self,  x):
        
        return self.__call__(x)
    

if __name__ == '__main__':
        
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# 0 = 打印所有信息（默认选项）
# 1 = 关闭 INFO 的打印
# 2 = 关闭 INFO 和 WARNING 的打印
# 3 = 关闭 INFO, WARNING, 和 ERROR 的打印

    # print(list(zip([1, 2, 3, 4], ['a', 'b', 'c', 'd'])))
    model = myModel(28*28,128,10)
    
    optimizer = optimizers.SGD()
    
    train_data, test_data = mnist_dataset()
    for epoch in range(50):
        loss, accuracy = model.train_one_step(optimizer, 
                                        tf.constant(train_data[0], dtype=tf.float32), 
                                        tf.constant(train_data[1], dtype=tf.int64))
        print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
    loss, accuracy = model.test(model, 
                          tf.constant(test_data[0], dtype=tf.float32), 
                          tf.constant(test_data[1], dtype=tf.int64))
    
    print('test loss', loss.numpy(), '; accuracy', accuracy.numpy())
    
    # 结果可视化
    idx = np.random.randint(1e4,size=9)
    x_test, y_test = test_data
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
    
    predictions = model.predict(tf.constant(x_test, dtype=tf.float32))
    y_pred = np.argmax(predictions, axis = 1)

plot_mnist_3_3(images, y_, y_pred[idx])

