# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/10/27 下午5:24
@summary:
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, Input, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CnnModel(Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 kernel_sizes=[2, 3, 5],
                 kernel_regularizer=None,
                 last_activation='softmax'
                 ):
        """
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param kernel_sizes: 滑动卷积窗口大小的list, eg: [2,3,5]
        :param kernel_regularizer: eg: tf.keras.regularizers.l2(0.001)
        :param class_num:
        :param last_activation:
        """
        super(CnnModel, self).__init__()
        self.maxlen = maxlen
        # self.max_features = max_features
        # self.embedding_dims = embedding_dims
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.conv1s = []
        self.avgpools = []
        for kernel_size in kernel_sizes:
            self.conv1s.append(Conv1D(filters=128, kernel_size=kernel_size, activation='relu', kernel_regularizer=kernel_regularizer))
            self.avgpools.append(GlobalMaxPooling1D())
        self.classifier = Dense(class_num, activation=last_activation, )

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextCNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextCNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i](emb)  # (batch_size, maxlen-kernel_size+1, filters)
            c = self.avgpools[i](c)  # # (batch_size, filters)
            conv1s.append(c)
        x = Concatenate()(conv1s)  # (batch_size, len(self.kernel_sizes)*filters)
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
        "kernel_sizes": [2, 3, 5]
    }
    # ============================Load Data==========================
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=config['max_features'])
    x_train = pad_sequences(x_train, maxlen=config['maxlen'])
    x_test = pad_sequences(x_test, maxlen=config['maxlen'])
    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        text_cnn = CnnModel(**config)
        model = text_cnn.build_graph()
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