# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/9/8 下午7:52
@summary:
"""
import jieba
import pandas as pd
base_path = '/data/python/data/THUCNews'

def load_data(file, out_file, train=True):
    """load data"""
    out_data = open(f"{base_path}/{out_file}", "w")
    df = pd.read_csv(f"{base_path}/{file}")

    count = 100000 if train else 10000
    i = 0
    for index, row in df.iterrows():
        print(i)
        with open(row['path'], 'r') as fr:
            text = fr.read()
        text = str(text.encode("utf-8"), 'utf-8')
        seg_text = jieba.cut(text.replace("\t", " ").replace("\n", " "))
        outline = " ".join(seg_text)
        outline = outline + "\t__label__" + str(row['label']) + "\n"

        out_data.write(outline)
        out_data.flush()
        i += 1
        if i > count:
            break
    out_data.close()


load_data('test.csv', 'test.txt', train=False)