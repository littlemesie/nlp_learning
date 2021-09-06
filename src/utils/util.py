# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/9/6 下午8:09
@summary:
"""

def load_vocab():
    """加载vocab数据"""
    word_dict = {}
    with open("../../data/dssm/vocab.txt", encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict