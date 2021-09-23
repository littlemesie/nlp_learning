# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/9/6 下午8:09
@summary:
"""
import pickle

def load_vocab():
    """加载vocab数据"""
    word_dict = {}
    with open("../../data/dssm/vocab.txt", encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict

def load_stopwords():
    """加载stopwords数据"""
    stopwords = set()
    with open("../../data/dssm/stopwords.txt", encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            stopwords.add(word)
    return stopwords


def save_pickle(data, filename):
    # 保存pickle文件
    with open(f"../../data/model_file/{filename}.pkl", 'wb') as f:
        pickle.dump(data, f)

def read_pickle(filename):
    # 读取pickle文件
    with open(f"../../data/model_file/{filename}.pkl", 'rb') as f:
        data = pickle.load(f)

    return data