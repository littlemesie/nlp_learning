# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/10/28 下午2:29
@summary: Text BIRNN
"""
# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/10/28 上午11:13
@summary: Text RNN 模型
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM, GRU, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from layers.modules import Fully


class BiRNN(Model):
    def __init__(self, maxlen, max_features, embedding_dims, class_num=5, dense_size=None,
                last_activation='softmax', **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        """
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param class_num:
        :param dense_size: fully layer 大小
        :param last_activation:
        """
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.dense_size = dense_size
        self.last_activation = last_activation

        self.embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims,
                                   input_length=self.maxlen)
        self.bi_rnn = Bidirectional(layer=GRU(units=128, activation='tanh', return_sequences=True), merge_mode='concat' ) # LSTM or GRU
        if self.dense_size is not None:
            self.ffn = Fully(self.dense_size)
        # self.avepool = GlobalAveragePooling1D()
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of TextBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        emb = self.embedding(inputs)

        x = self.bi_rnn(emb)
        # x = self.avepool(x)
        x = tf.reduce_mean(x, axis=1)
        if self.dense_size is not None:
            x = self.ffn(x)

        output = self.classifier(x)
        return output

    def build_graph(self, **kwargs):
        inputs = Input(shape=(self.maxlen), dtype=tf.int32)

        model = Model(inputs=inputs, outputs=self.call(inputs))
        return model

if __name__ == '__main__':
    """"""
    config = {
        "maxlen": 300,
        "max_features": 300,
        "embedding_dims": 128,
        "class_num": 2,
        "dense_size": [64, 32]
    }
    # ============================Load Data==========================
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=config['max_features'])
    x_train = pad_sequences(x_train, maxlen=config['maxlen'])
    x_test = pad_sequences(x_test, maxlen=config['maxlen'])
    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        text_birnn = BiRNN(**config)
        model = text_birnn.build_graph()
        model.summary()
        model.compile(optimizer=Adam(),
                     loss=SparseCategoricalCrossentropy(),
                     metrics=['accuracy'])

    # ============================Train Model==========================
    history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'valiation'], loc='upper left')
    plt.show()