# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/8/26 下午4:13
@summary: process data
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.util import load_vocab

path = '/data/python/data'

def process_text(x):
    """处理文本"""
    x = json.loads(x)
    return list(x.keys())

def preprocess_data():
    """load data"""
    names = ['keyword', 'text', 'click_title', 'category', 'label']
    df = pd.read_csv(f"{path}/train_100000.txt", sep='\t', names=names)

    df['text'] = df['text'].apply(lambda x: process_text(x))
    data_dict = {'keyword': [], 'doc': [], 'label': []}
    for index, row in df.iterrows():
        data_dict['keyword'].append(row['keyword'])
        data_dict['doc'].append(row['click_title'])
        data_dict['label'].append(1)
        texts = row['text'] if row['click_title'] not in row['text'] else row['text'].remove(row['click_title'])
        if not texts:
            continue
        for k in texts:
            data_dict['keyword'].append(row['keyword'])
            data_dict['doc'].append(k)
            data_dict['label'].append(0)
        # break

    data_df = pd.DataFrame.from_records(data_dict).reset_index(drop=True)
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    clounms = ['keyword', 'doc', 'label']
    data_df[clounms].to_csv(f"{path}/preprocess_train_100000.csv",  index=False)


def process_keyword(x, word_dict, key_len):
    """处理keyword"""
    words = list(x)[:key_len]
    if len(words) != key_len:
        words = words + ['[PAD]' for i in range(key_len - len(words))]
    xx = [word_dict[w] if w in word_dict else 0 for w in words]
    return xx


def process_doc(x, word_dict, doc_len):
    """处理keyword"""
    words = list(x)[:doc_len]
    if len(words) != doc_len:
        words = words + ['[PAD]' for i in range(doc_len - len(words))]
    xx = [word_dict[w] if w in word_dict else 0 for w in words]
    return xx

def get_feature(key_len=4, doc_len=10):
    """特征工程"""
    word_dict = load_vocab()
    df = pd.read_csv(f"{path}/preprocess_train_100000.csv")
    df['keyword_index'] = df['keyword'].apply(lambda x: process_keyword(x, word_dict, key_len))
    df['doc_index'] = df['doc'].apply(lambda x: process_doc(x, word_dict, doc_len))
    train_data, test_data = train_test_split(df, test_size=0.2)

    train_X = [train_data['keyword_index'].values, train_data['doc_index'].values]
    train_y = train_data['label'].values.astype('int32')

    test_X = [test_data['keyword_index'].values, test_data['doc_index'].values]
    test_y = test_data['label'].values.astype('int32')
    return word_dict, train_X, train_y, test_X, test_y

# preprocess_data()
# word_dict = load_vocab()
# print(word_dict)
# word_dict, train_X, train_y, test_X, test_y = get_feature()

