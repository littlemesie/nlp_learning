# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/11/16 上午10:20
@summary:
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()  # 用空格分开,所有的单词 有重复单词

word_list = " ".join(sentences).split()
word_list = list(set(word_list))  # 自动清除集合类型中的元素重复数据(set),以及元素排序
word_dict = {w: i for i, w in enumerate(word_list)}  # 编号,单词 : 编号

batch_size = 20  # To show 2 dim embedding graph
embedding_size = 2  # To show 2 dim embedding graph
voc_size = len(word_list)


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]])  # context word
        random_labels.append(data[i][1])   # target

    return random_inputs, random_labels

skip_grams = []
for i in range(1, len(word_sequence) - 1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
    for w in context:
        skip_grams.append([target, w])

input_batch, target_batch = random_batch(skip_grams, batch_size)
input_batch = np.array(input_batch, dtype=np.float32)
target_batch = np.array(target_batch, dtype=np.float32)

class Word2Vec(tf.keras.Model):
    def __init__(self):
        super(Word2Vec, self).__init__()

        self.W = tf.Variable(-2 * np.random.rand(voc_size, embedding_size) + 1, dtype=tf.float32)
        self.WT = tf.Variable(-2 * np.random.rand(embedding_size, voc_size) + 1, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        hidden_layer = tf.matmul(inputs, self.W)
        output_layer = tf.matmul(hidden_layer, self.WT)

        return output_layer


model = Word2Vec()
optimizer = tf.optimizers.Adam()
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
              metrics=['acc'])  # 记录参数,记录loss,正确率acc
# output = model(input_batch)
history = model.fit(input_batch, target_batch, epochs=10)

for i, label in enumerate(word_list):
    W, WT = model.variables
    x, y = float(W[i][0]), float(W[i][1])  # 第一列为x,第二列为y
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(6, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()

