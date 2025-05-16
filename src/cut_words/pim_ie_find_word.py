# -*- coding:utf-8 -*-

"""
@date: 2023/3/20 下午2:59
@summary: 基于互信息和左右熵
"""
import math
import re
from collections import Counter


def n_gram_words(text_list, n_gram):
    """
    获取 n_gram 的词频字典
    Args:
        text_list: 中文文章列表
        n_gram: 整数（int）类型的 n_gram 值
    Returns:
        返回词频字典
    """
    words = []
    for i in range(1, n_gram + 1):
        for text in text_list:
            words += [text[j:j + i] for j in range(len(text) - i + 1)]
    words_freq = dict(Counter(words))
    return words_freq

def pmi_filter(word_freq_dic, min_p):
    """
    获取符合 pmi 阈值的词
    Args:
        word_freq_dic: 词频字典
        min_p: PMI 阈值（最小值）
    Returns:
        返回符合条件的词列表
    """
    new_words = []
    for word in word_freq_dic:
        if len(word) == 1:
            #   单个字，跳过忽略
            pass
        else:
            #   p(x)*p(y)，根据 PMI 公式，该部分越小，PMI 值越大，所以取最小值即可表示该 word 的最大 PMI 值
            p_x_y = min([word_freq_dic.get(word[:i]) * word_freq_dic.get(word[i:]) for i in range(1, len(word))])
            pmi = math.log2((word_freq_dic.get(word) * len(word_freq_dic)) / p_x_y)
            #   也可以不计算 log2，加快统计速度，这时阈值需要新的量纲设置
            #   pmi = (word_freq_dic.get(word) * len(word_freq_dic)) / p_x_y
            if pmi > min_p:
                new_words.append(word)
    return new_words

def calculate_entropy(char_list):
    """
    计算 char_list 的信息熵
    Args:
        char_list: 出现词的列表
    Returns:
        出现词的信息熵
    """
    char_freq_dic = dict(Counter(char_list))
    entropy = (-1) * sum(
        [char_freq_dic.get(i) / len(char_list) * math.log2(char_freq_dic.get(i) / len(char_list)) for i in
         char_freq_dic])
    return entropy


def entropy_left_right_filter(conditional_words, text_list, ngram, min_entropy):
    """
    在符合 PMI 过滤条件的词中，获取符合最小信息熵条件的最终候选新词
    Args:
        conditional_words: 符合 PMI 过滤条件的词的列表
        text_list: 原始文本列表
        ngram: 指定的 ngram
        min_entropy: 最小信息熵，符合该条件才可以作为候选新词
    Returns:
        最终的候选词，字典（dict）类型
    """
    final_words = {}
    for word in conditional_words:
        left_char_list = []
        right_char_list = []
        for text in text_list:
            left_right_char_tuple_list = re.findall('(?:(.{0,%d}))%s(?=(.{0,%d}))' % (ngram, word, ngram), text)
            if left_right_char_tuple_list:
                for left_right_char_tuple in left_right_char_tuple_list:
                    if left_right_char_tuple[0]:
                        left_char_list += [left_right_char_tuple[0][i:] for i in range(len(left_right_char_tuple[0]))]
                    if left_right_char_tuple[1]:
                        right_char_list += [left_right_char_tuple[1][i:] for i in range(len(left_right_char_tuple[1]))]
        if left_char_list:
            left_entropy = calculate_entropy(left_char_list)
        if right_char_list:
            right_entropy = calculate_entropy(right_char_list)
        if left_char_list and right_char_list:
            if min(right_entropy, left_entropy) > min_entropy:
                final_words[word] = min(right_entropy, left_entropy)
        elif left_char_list:
            if left_entropy > min_entropy:
                final_words[word] = left_entropy
        elif right_char_list:
            if right_entropy > min_entropy:
                final_words[word] = right_entropy
    sorted(final_words.items(), key=lambda x: x[1], reverse=False)
    return final_words