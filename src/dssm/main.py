# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/9/30 下午2:02
@summary:
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from dssm.process_data import get_feature
from dssm.model import Dssm

if __name__ == '__main__':
    config = {
        'query_max_len': 4,
        'doc_max_len': 10,
        'embedding_size': 256,
        'vec_dim': 256
    }

    word_dict, train_X, train_y, test_X, test_y = get_feature(config['query_max_len'], config['doc_max_len'])
    config.update({'vocab_size': len(word_dict)})
    print(train_X)
    dssm = Dssm(**config)
    model = dssm.build_model()
    model.summary()
    model.compile(optimizer='sgd',
                  loss={'out': 'binary_crossentropy'},
                  metrics={"out": [Accuracy(), Precision(), Recall()]})
    # model.fit(
    #     train_X,
    #     train_y,
    #     epochs=5,
    #     # callbacks=[EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint
    #     batch_size=256,
    #     validation_split=0.1
    # )