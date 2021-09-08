# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/8/31 下午8:08
@summary:
"""
import gensim
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def train(basename):
    # 定义输入输出
    inp = basename + 'wiki.zh.text'
    outp1 = basename + 'wiki.zh.text.model'
    outp2 = basename + 'wiki.zh.text.vector'

    model = Word2Vec(LineSentence(inp), vector_size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)


if __name__ == '__main__':
    """"""
    # 定义输入输出
    basename = "/data/python/data/"
    model_path = basename + 'wiki.zh.text.model'
    # train(basename)

    # 测试
    model = gensim.models.Word2Vec.load(model_path)

    result = model.wv.similar_by_word(word='富士', topn=10)
    print(result)

