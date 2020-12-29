# -*- coding: utf-8 -*-

"""
Created on 12/29/20 6:13 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

尝试通过词袋模型+线性模型完成文本分类。
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

if __name__ == '__main__':
    train_df = pd.read_csv('/Users/chiang/数据/天池_零基础入门NLP_新闻文本分类/train_set.csv', sep='\t', nrows=15000)
    vectorizer = CountVectorizer(max_features=3000)
    train_test = vectorizer.fit_transform(train_df['text'])  # 将文本中的词语转换为词频矩阵

    print(train_test[0])  # 在输出中，左边的括号中的第一个数字是文本的序号i，第2个数字是词的序号j，注意词的序号是基于所有的文档的。第三个数字就是我们的词频。
    clf = RidgeClassifier()
    clf.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = clf.predict(train_test[10000:])
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))  # 0.741494277019762
