# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/11/16 上午11:14
@summary: skip grams model
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from layers.modules import SampledSoftmaxLayer
from utils.loss_util import sampledsoftmaxloss

def generate_data():
    """生成数据"""
    sentences = ["i like dog", "i like cat", "i like animal",
                 "dog cat animal", "apple cat dog like", "dog fish milk like",
                 "dog cat eyes like", "i like apple", "apple i hate",
                 "apple i movie book music like", "cat dog hate", "cat dog like"]

    word_sequence = " ".join(sentences).split()  # 用空格分开,所有的单词 有重复单词

    word_list = list(set(word_sequence))  # 自动清除集合类型中的元素重复数据(set),以及元素排序
    word_dict = {w: i for i, w in enumerate(word_list)}  # 编号,单词 : 编号
    voc_size = len(word_list)

    return word_sequence, word_dict, voc_size

def generate_train_data(skip_grams, voc_size):
    train_inputs = []
    train_labels = []
    for d in skip_grams:
        train_inputs.append(d[0])
        # train_inputs.append(np.eye(voc_size)[d[0]])  # target
        train_labels.append(d[1])   # context word

    return train_inputs, train_labels

def generate_skip_grams(word_sequence, word_dict):
    """generate skip grams"""
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        skip_grams.append([target, context])
        # for w in context:
        #     skip_grams.append([target, w])

    return skip_grams

class Word2Vec(Model):
    def __init__(self, voc_size, embedding_size, num_sampled=5):
        super(Word2Vec, self).__init__()
        self.voc_size = voc_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.sampled_softmax = SampledSoftmaxLayer(num_sampled=self.num_sampled)

    def call(self, inputs, **kwargs):
        train_inputs, labels_inputs = inputs
        print(train_inputs.shape)
        print(labels_inputs.shape)
        train_inputs = tf.squeeze(train_inputs, axis=1)
        labels_inputs = tf.squeeze(labels_inputs, axis=1)
        print(train_inputs.shape)
        print(labels_inputs.shape)
        embeddings = tf.Variable(tf.random.uniform([voc_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        print(embeddings.shape)
        print(embed.shape)
        output = self.sampled_softmax([embeddings, embed, labels_inputs])

        return output

    def build_graph(self, **kwargs):
        """"""
        train_inputs = Input(shape=(None, ), dtype=tf.int32)
        labels_inputs = Input(shape=(None, None), dtype=tf.int32)
        model = Model(inputs=[train_inputs, labels_inputs],
                      outputs=self.call([train_inputs, labels_inputs]))
        return model

if __name__ == '__main__':

    embedding_size = 16  # To show 2 dim embedding graph
    learning_rate = 0.001
    word_sequence, word_dict, voc_size = generate_data()
    skip_grams = generate_skip_grams(word_sequence, word_dict)
    train_inputs, train_labels = generate_train_data(skip_grams, voc_size)
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)

    print([train_inputs, train_labels])
    w2c = Word2Vec(voc_size, embedding_size)
    model = w2c.build_graph()
    model.summary()
    # ============================Compile============================
    model.compile(optimizer=Adam(), loss=sampledsoftmaxloss,  metrics=['acc'])
    model.fit([train_inputs, train_labels], train_labels, epochs=10)
    # a = tf.Variable(-2 * np.random.rand(voc_size, embedding_size) + 1, dtype=tf.float32)
    # print(voc_size)
    # print(train_inputs.shape)
    # print(a.shape)

    # embeddings = tf.Variable(tf.random.uniform([voc_size, embedding_size], -1.0, 1.0))
    # print(train_labels)
    # print(embeddings.shape)
    # train_inputs = tf.convert_to_tensor(np.array(train_inputs))
    # # print(train_inputs)
    # embed = tf.nn.embedding_lookup(embeddings, train_labels)
    # print(embed)