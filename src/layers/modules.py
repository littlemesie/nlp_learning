# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/8/26 下午4:10
@summary:
"""
from tensorflow.keras.layers import Layer, Dense, Dropout

class DNN(Layer):
    """DNN Layer"""
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
        """
        DNN part
        :param hidden_units: A list. List of hidden layer units's numbers
        :param activation: A string. Activation function
        :param dnn_dropout: A scalar. dropout number
        """
        super(DNN, self).__init__(**kwargs)
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x