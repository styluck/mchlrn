# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:32:00 2024

@author: lich5
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import numpy as np
import os

# poem_type 为 0 表示整体生成，为 1 表示生成藏头诗
def generate_text(model, start_string, poem_type):
    # 控制诗句意境
    prefix_words = '月上柳梢头，人约黄昏后。'
    # 要生成的字符个数
    num_generate = 120
    # 空字符串用于存储结果
    poem_generated = []
    temperature = 1.0
    # 以开头正常生成
    if poem_type == 0:
        # 将整个输入直接导入
        input_eval = [word2ix[s] for s in prefix_words + start_string]
        # 添加开始标识
        input_eval.insert(0, word2ix['<START>'])
        input_eval = tf.expand_dims(input_eval, 0)
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # 删除批次的维度
            predictions = tf.squeeze(predictions, 0)
            # 用分类分布预测模型返回的字符
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions,
                                                 num_samples=1)[-1, 0].numpy()
            # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
            input_eval = tf.expand_dims([predicted_id], 0)
            poem_generated.append(ix2word[predicted_id])
        # 删除多余的字
        try:
            del poem_generated[poem_generated.index('<EOP>'):]
        except:
            pass
        
        return (start_string + ''.join(poem_generated))
    # 藏头诗
    if poem_type == 1:
        for i in range(len(start_string)):
            # 藏头诗以每个字分别生成诗句
            input_eval = [word2ix[s] for s in prefix_words + start_string[i]]
            input_eval.insert(0, word2ix['<START>'])
            input_eval = tf.expand_dims(input_eval, 0)
            model.reset_states()
            poem_one = [start_string[i]]
            for j in range(num_generate):
                predictions = model(input_eval)
                # 删除批次的维度
                predictions = tf.squeeze(predictions, 0)
                # 用分类分布预测模型返回的字符
                predictions = predictions / temperature
                predicted_id = tf.random.categorical(
                    predictions, num_samples=1)[-1, 0].numpy()
                # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
                input_eval = tf.expand_dims([predicted_id], 0)
                poem_one.append(ix2word[predicted_id])
            # 删除多余的字
            try:
                del poem_one[poem_one.index('。') + 1:]
            except:
                pass
            poem_generated.append(''.join(poem_one) + '\n')
        return ''.join(poem_generated)
    
# 切分成输入和输出
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# 损失函数
def loss(labels, logits):
    '''	
    from_logits=False (default)
    Whether y_pred is expected to be a logits tensor. 
    '''
    return losses.sparse_categorical_crossentropy(labels,
                                                  logits,
                                                  from_logits=True)

if __name__ == '__main__':

    # 导入数据集
    data = np.load('data.npy', allow_pickle=True).tolist()
    data_line = np.array([word for poem in data for word in poem])
    ix2word = np.load('ix2word.npy', allow_pickle=True).item()
    word2ix = np.load('word2ix.npy', allow_pickle=True).item()
    
    
    # 每批大小
    batch_size = 64
    # 缓冲区大小
    BUFFER_SIZE = 10000
    # 训练周期
    EPOCHS = 20
    # 诗的长度
    poem_size = 125
    # 嵌入的维度
    embedding_dim = 64
    # RNN 的单元数量
    units = 128
    
    vocab_size=len(ix2word)
    
    # 创建训练样本
    poem_dataset = tf.data.Dataset.from_tensor_slices(data_line)
    # 将每首诗提取出来并切分成输入输出
    poems = poem_dataset.batch(poem_size + 1, drop_remainder=True)
    dataset = poems.map(split_input_target)
    # 分批并随机打乱
    dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
        
    # 构建RNN模型
    model = keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim))
    model.add(layers.SimpleRNN(units, return_sequences=True))
    model.add(layers.SimpleRNN(units, return_sequences=True))
    model.add(layers.SimpleRNN(units, return_sequences=True))
    model.add(layers.Dense(vocab_size))
    # 构建GRU模型
    # model = keras.Sequential()
    # model.add(layers.Embedding(vocab_size, embedding_dim))
    # model.add(layers.GRU(units, return_sequences=True, stateful=True,
    #                      recurrent_initializer='glorot_uniform'))
    # model.add(layers.GRU(units, return_sequences=True, stateful=True,
    #                      recurrent_initializer='glorot_uniform'))
    # model.add(layers.Dense(vocab_size))
    
    # 构建LSTM模型
    # model = keras.Sequential()
    # model.add(layers.Embedding(vocab_size, embedding_dim))
    # model.add(layers.LSTM(units, dropout=0.5, return_sequences=True,
    #                       recurrent_initializer='glorot_uniform'))
    # model.add(layers.LSTM(units, dropout=0.5, return_sequences=True,
    #                       recurrent_initializer='glorot_uniform'))
    # model.add(layers.Dense(vocab_size))
    
    # 构建双向循环模型
    # model = keras.Sequential()
    # model.add(layers.Embedding(vocab_size, embedding_dim))
    # model.add(layers.Bidirectional(layers.LSTM(units, dropout=0.5, return_sequences=True,
    #                       recurrent_initializer='glorot_uniform')))
    # model.add(layers.LSTM(units, dropout=0.5, return_sequences=True,
    #                       recurrent_initializer='glorot_uniform'))
    # model.add(layers.Dense(vocab_size))
    
    optimizer = keras.optimizers.Adam(clipnorm=1.)
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    
    # 设置检查点的目录
    checkpoint_dir = './training_checkpoints_tmp'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    # 用于在训练过程中保存模型。这个回调允许你根据训练的进度定期保存
    # 模型的参数（weights）和优化器状态（optimizer state）
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True)
    
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # 需要时，直接载入之前保存在checkpoint的模型
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    model.summary()
    
    # model.build(tf.TensorShape([1, None]))

    # 生成以"水"开头的诗词
    print(generate_text(model, start_string="水", poem_type=0))
    
    # 生成以"水墨云烟"开头的藏头诗
    print(generate_text(model, start_string="水墨云烟", poem_type=1))
    
    # [EOF]
