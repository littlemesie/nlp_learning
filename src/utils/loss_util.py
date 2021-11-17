# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/11/17 下午5:24
@summary:
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K

def sampledsoftmaxloss(y_pred):

    return tf.reduce_mean(y_pred)
