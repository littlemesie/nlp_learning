# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/9/9 上午9:41
@summary:
"""
import fasttext
base_path = '/data/python/data/THUCNews'
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