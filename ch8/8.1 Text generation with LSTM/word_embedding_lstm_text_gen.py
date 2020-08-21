"""
基于词嵌入的 LSTM 文本生成
"""

import random
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras
import numpy as np
import jieba
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

path = '/Users/c/Desktop/b.txt'
text = open(path).read().lower().replace('\n', '').replace('　　', '\n')
print('Corpus length:', len(text))

# 将文本序列向量化

maxlen = 60     # 每个序列的长度
step = 3        # 每 3 个 token 采样一个新序列
sentences = []  # 保存所提取的序列
next_tokens = []  # sentences 的下一个 token

print('Vectorization...')

token_text = list(jieba.cut(text))

tokens = list(set(token_text))
tokens_indices = {token: tokens.index(token) for token in tokens}
print('Number of tokens:', len(tokens))

for i in range(0, len(token_text) - maxlen, step):
    sentences.append(
        list(map(lambda t: tokens_indices[t], token_text[i: i+maxlen])))
    next_tokens.append(tokens_indices[token_text[i+maxlen]])
print('Number of sequences:', len(sentences))

next_tokens_one_hot = []
for i in next_tokens:
    y = np.zeros((len(tokens),), dtype=np.bool)
    y[i] = 1
    next_tokens_one_hot.append(y)

dataset = tf.data.Dataset.from_tensor_slices((sentences, next_tokens_one_hot))
dataset = dataset.shuffle(buffer_size=4096)
dataset = dataset.batch(128)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# 构建模型

model = models.Sequential([
    layers.Embedding(len(tokens), 256),
    layers.LSTM(256),
    layers.Dense(len(tokens), activation='softmax')
])

# 模型编译配置

optimizer = optimizers.RMSprop(lr=0.1)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)

# 采样函数


def sample(preds, temperature=1.0):
    '''
    对模型得到的原始概率分布重新加权，并从中抽取一个 token 索引
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 训练模型


callbacks_list = [
    # 在每轮完成后保存权重
    keras.callbacks.ModelCheckpoint(
        filepath='text_gen.h5',  # 保存文件的路径
        monitor='loss',      # monitor：要验证的指标
        save_best_only=True,     # 只保存让 monitor 指标最好的模型（如果 monitor 没有改善，就不保存）
    ),
    # 不再改善时降低学习率
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss',    # 要验证的指标
        factor=0.5,            # 触发时：学习率 *= factor
        patience=1,            # monitor 在 patience 轮内没有改善，则触发降低学习率
    ),
    # 不再改善时中断训练
    keras.callbacks.EarlyStopping(
        monitor='loss',           # 要验证的指标
        patience=3,             # 如果 monitor 在多于 patience 轮内（比如这里就是10+1=11轮）没有改善，则中断训练
    ),
]

model.fit(dataset, epochs=60, callbacks=callbacks_list)

# 文本生成

start_index = random.randint(0, len(text) - maxlen - 1)
generated_text = text[start_index: start_index + maxlen]
# print(f' Generating with seed: "{generated_text}"')
print(f'  📖 Generating with seed: "\033[1;32;43m{generated_text}\033[0m"')

for temperature in [0.2, 0.5, 1.0, 1.2]:
    # print('\n  temperature:', temperature)
    print(f'\n   \033[1;36m 🌡️ temperature: {temperature}\033[0m')
    print(generated_text, end='')
    for i in range(400):    # 生成 400 个 token
        # 编码当前文本
        text_cut = jieba.cut(generated_text)
        sampled = []
        for i in text_cut:
            if i in tokens_indices:
                sampled.append(tokens_indices[i])
            else:
                sampled.append(0)

        # 预测，采样，生成下一个 token
        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_token = tokens[next_index]
        print(next_token, end='')

        generated_text = generated_text[1:] + next_token
