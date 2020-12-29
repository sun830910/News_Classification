# -*- coding: utf-8 -*-

"""
Created on 12/29/20 6:36 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
尝试TF-IDF + RidgeClassifier的文本分类
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

if __name__ == '__main__':
    train_df = pd.read_csv('/Users/chiang/数据/天池_零基础入门NLP_新闻文本分类/train_set.csv', sep='\t', nrows=15000)

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
    train_test = tfidf.fit_transform(train_df['text'])

    clf = RidgeClassifier()
    clf.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = clf.predict(train_test[10000:])
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))
