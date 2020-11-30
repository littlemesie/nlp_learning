#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @version: V-17-5-2
    @author: Linlifang
    @file: crf_model.py
    @time: 17-5-2.下午1:37
"""
import os
import subprocess
import CRFPP
import jieba

class CRFModel(object):
    def __init__(self, model='model_name'):
        """
        函数说明: 类初始化
        :param model: 模型名称
        """
        self.model = model

    def add_tagger(self, tag_data):
        """
        函数说明: 添加语料
        :param tag_data: 数据
        :return:
        """
        word_str = jieba.cut(tag_data)

        if not os.path.exists(self.model):
            print('模型不存在,请确认模型路径是否正确!')
            exit()
        tagger = CRFPP.Tagger("-m {} -v 3 -n2".format(self.model))
        tagger.clear()
        for word in word_str:
            print(word)
            tagger.add(word)
        tagger.parse()
        return tagger

    def text_mark(self, tag_data, begin='B', middle='I', end='E', single='S'):
        """
        文本标记
        :param tag_data: 数据
        :param begin: 开始标记
        :param middle: 中间标记
        :param end: 结束标记
        :param single: 单字结束标记
        :return result: 标记列表
        """
        tagger = self.add_tagger(tag_data)
        print(tagger)
        size = tagger.size()
        print(size)
        tag_text = ""
        for i in range(0, size):
            word, tag = tagger.x(i, 0), tagger.y2(i)
            print(tag)
            if tag in [begin, middle]:
                tag_text += word
            elif tag in [end, single]:
                tag_text += word + "*&*"
        result = tag_text.split('*&*')
        result.pop()
        return result

    def crf_test(self, tag_data, separator='_'):
        """
        函数说明: crf测试
        :param tag_data:
        :param separator:
        :return:
        """
        result = self.text_mark(tag_data)
        print(result)
        data = separator.join(result)
        return data

    def crf_learn(self, filename):
        """
        函数说明: 训练模型
        :param filename: 已标注数据源
        :return:
        """
        crf_bash = "crf_learn -f 3 -c 4.0 api/template.txt {} {}".format(filename, self.model)
        process = subprocess.Popen(crf_bash.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]
        print(output.decode(encoding='utf-8'))


if __name__ == '__main__':
    """"""
    # crf_model = CRFModel(model='model')
    # a = '龙骧短柄翻盖旅行包'
    # data = crf_model.crf_test(tag_data=a)
    # print(data)
    # word_str = jieba.cut("龙骧短柄翻盖旅行包", cut_all=False)
    # tagger = CRFPP.Tagger("-m model -v 3 -n2")
    # tagger.clear()
    # for w in word_str:
    #     print(w)
    #     tagger.add(w)
    # tagger.add("龙 骧")
    # tagger.add("短 柄")
    # tagger.add("翻 盖")
    # tagger.add("旅 行 包")
    # tagger.add("Confidence NN")
    # tagger.add("in IN")
    # tagger.add("the DT")
    # tagger.add("pound NN")
    # tagger.add("is VBZ")
    # tagger.add("widely RB")
    # tagger.parse()
    # print("column size: ", tagger.xsize())
    # print("token size: ", tagger.size())
    # print("tag size: ", tagger.ysize())
    filename = 'crf_test.data'
    crf_bash = "crf_test -m model {} ".format(filename)
    process = subprocess.Popen(crf_bash.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]
    for o in output.decode(encoding='utf-8'):
        print(o)
    # print(output.decode(encoding='utf-8'))
