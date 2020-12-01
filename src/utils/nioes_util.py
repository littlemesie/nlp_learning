import re
import jieba
import random

s_list = []
bow_list = []

synonym_dict = {}
color_dict = {}
product_dict = {}
brand_dict = {}
location_dict = {}
material_dict = {}
style_dict = {}
sex_dict = {}
effect_dict = {}

def load_synonym_dict():
    with open("../../data/entity/synonym_ext.dic", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            synonym_dict[line[0]] = line[1]
    return

def read_color_dict():
    with open("../../data/entity/color_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            color_dict[line] = 1
    return

def read_product_dict():
    with open("../../data/entity/product_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            product_dict[line] = 1
    return

def read_brand_dict():
    with open("../../data/entity/brand_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            brand_dict[line.lower()] = 1
    return

def read_location_dict():
    with open("../../data/entity/location_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            location_dict[line] = 1
    return

def read_material_dict():
    with open("../../data/entity/material_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            material_dict[line] = 1
    return

def read_style_dict():
    with open("../../data/entity/style_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            style_dict[line] = 1
    return

def read_sex_dict():
    with open("../../data/entity/sex_ext.dic", "r") as f:
        for line in f:
            line = line.strip()
            sex_dict[line] = 1
    return

def read_effect_dict():
    with open("../../data/entity/effect_ext.dict", "r") as f:
        for line in f:
            line = line.strip()
            effect_dict[line] = 1
    return

def clean_and():
    jieba.load_userdict("../../data/entity/segmention/unigram.txt")
    train_output = open("../../data/named_entity_recognition/train_taobao.data", "w")
    test_output = open("../../data/named_entity_recognition/test_taobao.data", "w")
    with open("../../data/entity/prepare_data", "r") as f:
        for line in f:
            line = "".join(line.split())
            # 大小写归一化
            line = line.lower()

            # 清除过短的query
            if len(line) <= 2:
                continue

            # 清除爬虫爬宝贝id的query
            if re.match('[0-9]{18}', line) != None:
                continue

            # 过滤全是英文的query
            eng_flag = True
            for i in line:
                if i >= u'\u4e00' and i <= u'\u9fa5':
                    eng_flag = False
                    break
            if eng_flag == True:
                continue

            # 重新分词
            ll = jieba.cut(line)
            line = []
            for i in ll:
                if i == u"\u2006" or i == u" " or i == " ":
                    continue
                line.append(i)

            # 同义词替换，简写替换
            for i in range(len(line)):
                if synonym_dict.get(line[i], None):
                    line[i] = synonym_dict[line[i]]

            # 过滤重复query
            if line in s_list:
                continue

            s_list.append(line)
            bioes_list = bioes_label(line)
            if random.random() < 0.8:
                for bl in bioes_list:
                    l = " ".join(bl)
                    # print(l)
                    train_output.write(l + "\n")

                print(line)
                train_output.write("\n")
            else:
                for bl in bioes_list:
                    l = " ".join(bl)
                    # print(l)
                    test_output.write(l + "\n")

                print(line)
                test_output.write("\n")

    train_output.close()
    test_output.close()
    return

def is_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def bioes_label(line):
    """BIOES标注"""
    bioes_list = []
    for term in line:
        if color_dict.get(term, None):
            tag = "color"
        elif product_dict.get(term, None):
            tag = "product"
        elif brand_dict.get(term, None):
            tag = "brand"
        elif location_dict.get(term, None):
            tag = "location"
        elif material_dict.get(term, None):
            tag = "material"
        elif style_dict.get(term, None):
            tag = "style"
        elif sex_dict.get(term, None):
            tag = "sex"
        elif effect_dict.get(term, None):
            tag = "effect"
        else:
            tag = 'other'
        bioes_list = is_bioes(term, tag, bioes_list)
    return bioes_list


def is_bioes(term, tag, bioes_list):

    if is_chinese(term):
        term = list(term.strip())
        if len(term) == 1:
            flag = 'S-{}'.format(tag)
            bioes_list.append((term[0], flag))
        else:
            for i, word in enumerate(term):
                word = "".join(word.split())
                if not word:
                    continue
                if i == 0:
                    flag = 'B-{}'.format(tag)
                    bioes_list.append((word, flag))
                elif i == len(term) - 1:
                    flag = 'E-{}'.format(tag)
                    bioes_list.append((word, flag))
                else:
                    flag = 'I-{}'.format(tag)
                    bioes_list.append((word, flag))
    else:
        flag = 'S-{}'.format(tag)
        bioes_list.append((term, flag))


    return bioes_list

read_color_dict()
read_product_dict()
read_brand_dict()
read_location_dict()
read_material_dict()
read_style_dict()
read_sex_dict()
read_effect_dict()
# line = ['aape', '上衣']
# line = ['abybom', '艾柏', '梵', '超能', '婴儿', '桃花', '面膜', '基因', '再生', '补水', '保湿']
# line = ['魔法森林', '施华洛', '奇', '戒指']
# bioes_label(line)
clean_and()
a = '  '
