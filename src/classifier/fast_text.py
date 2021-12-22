# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/12/22 下午6:49
@summary:
"""
import jieba
import fasttext
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


# load_data('test.csv', 'test.txt', train=False)

# 训练模型
# classifier = fasttext.train_supervised(f"{base_path}/train.txt", label_prefix="__label__")
# classifier.save_model(f"{base_path}/model.bin")

# 加载模型
classifier = fasttext.load_model(f"{base_path}/model.bin")
# 测试模型
# result = classifier.test(f"{base_path}/test.txt")
# print('precision：', result[1])
labels_right = []
texts = []
with open(f"{base_path}/test.txt") as fr:
    for line in fr:
        line = str(line.encode("utf-8"), 'utf-8').rstrip()
        labels_right.append(line.split("\t")[1].replace("__label__",""))
        texts.append(line.split("\t")[0])

labels_predict = [term[0] for term in classifier.predict(texts)[0]] #预测输出结果为二维形式
print(labels_right)
print(labels_predict)