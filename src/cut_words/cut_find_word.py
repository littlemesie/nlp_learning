# -*- coding:utf-8 -*-

"""
@date: 2023/2/21 下午5:47
@summary: 通过切分新词发现
"""

import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from math import log
import re

class Find_Words:
    def __init__(self, min_count=10, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int)  # 如果键不存在，那么就用int函数初始化一个值，int()的默认结果为0
        self.total = 0.
    def text_filter_(self, texts):
        # 预切断句子，以免得到太多无意义（不是中文、英文、数字）的字符串
        for a in tqdm(texts):
            # 这个正则表达式匹配的是任意非中文、
            # 非英文、非数字，因此它的意思就是用任
            # 意非中文、非英文、非数字的字符断开句子
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', a):
                if t:
                    yield t
    def text_filter(self):
        path = ""
        df = pd.read_csv(path)
        df = df.sample(10000)
        # 预切断句子，以免得到太多无意义（不是中文、英文、数字）的字符串
        for _, row in df.iterrows():
            # 这个正则表达式匹配的是任意非中文、
            # 非英文、非数字，因此它的意思就是用任
            # 意非中文、非英文、非数字的字符断开句子
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', row['text']):
                if t:
                    yield t

    def count(self):
        # 计数函数，计算单字出现频数、相邻两字出现的频数
        for text in self.text_filter():
            self.chars[text[0]] += 1
            for i in range(len(text)-1):
                self.chars[text[i+1]] += 1
                self.pairs[text[i:i+2]] += 1
                self.total += 1
        self.chars = {i: j for i, j in self.chars.items() if j >= self.min_count}  # 最少频数过滤
        self.pairs = {i: j for i, j in self.pairs.items() if j >= self.min_count}  # 最少频数过滤
        self.strong_segments = set()
        for i, j in self.pairs.items():  # 根据互信息找出比较“密切”的邻字
            _ = log(self.total*j/(self.chars[i[0]]*self.chars[i[1]]))
            if _ >= self.min_pmi:
                self.strong_segments.add(i)

    def find_words(self):
        # 根据前述结果来找词语
        self.words = defaultdict(int)
        for text in self.text_filter():
            s = text[0]
            print(text)
            for i in range(len(text)-1):
                if text[i:i+2] in self.strong_segments:  # 如果比较“密切”则不断开
                    s += text[i+1]
                else:
                    self.words[s] += 1  # 否则断开，前述片段作为一个词来统计
                    s = text[i+1]
            self.words[s] += 1  # 最后一个“词”
        self.words = {i: j for i, j in self.words.items() if j >= self.min_count}  # 最后再次根据频数过滤

# fw = Find_Words(2, 1)
# fw.count()
# fw.find_words()
# words = pd.Series(fw.words).sort_values(ascending=False)
# print(words)

