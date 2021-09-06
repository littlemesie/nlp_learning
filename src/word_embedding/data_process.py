# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/8/31 下午7:52
@summary:
"""
import os.path
import sys
from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    # 定义输入输出
    basename = "/data/python/data/"
    inp = basename + 'zhwiki-latest-pages-articles.xml.bz2'
    outp = basename + 'wiki.zh.text'
    space = " "
    i = 0
    output = open(outp, 'w', encoding='utf-8')
    wiki = WikiCorpus(inp, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            print("Saved " + str(i) + " articles")
    output.close()
    print("Finished Saved " + str(i) + " articles")