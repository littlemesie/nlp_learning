# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/8/26 下午3:51
@summary: DSSM Model
"""

import os
import numpy as np
import faiss
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Input, Lambda, Dense, Dot
from layers.modules import DNN

class Dssm(Model):

    def __init__(self, query_max_len, doc_max_len, vocab_size, embedding_size=128, vec_dim=64, dnn_activation='relu',
               l2_reg_embedding=1e-6, dnn_dropout=0.0, **kwargs):
        super(Dssm, self).__init__(**kwargs)
        self.query_max_len = query_max_len
        self.doc_max_len = doc_max_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.vec_dim = vec_dim


    def call(self, inputs, training=None, mask=None):
        query, doc = inputs
        embedding = Embedding(self.vocab_size, self.embedding_size, name='embedding')
        # average embedding
        avg_embedding = Lambda(lambda x: tf.reduce_mean(x, 1))

        query_embed = embedding(query)
        query_embed = avg_embedding(query_embed)
        doc_embed = embedding(doc)
        doc_embed = avg_embedding(doc_embed)

        query_dense0 = Dense(1024)(query_embed)
        query_vec = Dense(self.vec_dim, name='query_vec')(query_dense0)

        doc_dense0 = Dense(1024)(doc_embed)
        doc_vec = Dense(self.vec_dim, name='doc_vec')(doc_dense0)

        cos = Dot(axes=-1, normalize=True, name='cosine')([query_vec, doc_vec])
        out = Dense(1, activation='sigmoid', name='out')(cos)

        return {'out': out, 'cosine': cos}

    def build_model(self, **kwargs):
        query = Input(shape=(self.query_max_len,), dtype=tf.int64, name='query_input')
        doc = Input(shape=(self.doc_max_len,), dtype=tf.int64, name='doc_input')

        model = Model(inputs=[query, doc], outputs=self.call([query, doc]))

        return model
