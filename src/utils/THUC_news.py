# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/9/2 下午7:50
@summary: THUCNews data process
"""
import os
import random
import pandas as pd
base_path = '/data/python/data/THUCNews'

def generate_map_file():
    """
    生成映射文件
    """
    labels = {'财经': 1, '彩票': 2, '房产': 3, '股票': 4, '家居': 5, '教育': 6, '科技': 7, '社会': 8, '时尚': 9, '时政': 10,
              '体育': 11, '星座': 12, '游戏': 13, '娱乐': 14}

    train_index_map = {'path': [], 'label': []}
    test_index_map = {'path': [], 'label': []}

    for key, label in labels.items():
        files_path = f"{base_path}/{key}"
        file_list = os.listdir(files_path)
        for file in file_list:
            rate = random.random()
            if rate <= 0.8:
                train_index_map['path'].append(f"{files_path}/{file}")
                train_index_map['label'].append(label)
            else:
                test_index_map['path'].append(f"{files_path}/{file}")
                test_index_map['label'].append(label)
    train_df = pd.DataFrame.from_records(train_index_map).sample(frac=1).reset_index(drop=True)
    test_df = pd.DataFrame.from_records(test_index_map).sample(frac=1).reset_index(drop=True)
    train_df.to_csv(f"{base_path}/train.csv", index=False)
    test_df.to_csv(f"{base_path}/test.csv", index=False)
    print(train_df)
    print(test_df)

# generate_map_file()