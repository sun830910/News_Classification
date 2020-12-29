# -*- coding: utf-8 -*-

"""
Created on 12/29/20 2:11 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import pandas as pd
from collections import Counter


def length_analysis(df):
    df['text_len'] = df['text'].apply(lambda x: len(x.split(' ')))
    print(df['text_len'].describe())


def word_freq_analysis(df):
    all_lines = ' '.join(list(df['text']))
    word_count = Counter(all_lines.split(" "))
    word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
    print("训练集中共有 {} 个不同的字".format(len(word_count)))
    print("训练集中出现最多的字是{word},共出现{cnt}次".format(word=word_count[0][0], cnt=word_count[0][1]))
    print("训练集中出现最少的字是{word},共出现{cnt}次".format(word=word_count[-1][0], cnt=word_count[-1][1]))


def label_freq_analysis(df):
    label_record = dict()
    labels = df['label']
    for label in labels:
        if label not in label_record:
            label_record.setdefault(label, 1)
        else:
            label_record[label] += 1
    df_length = len(labels)
    print("共有{}个样本".format(df_length))
    for key in label_record.keys():
        print("第{}个类别共有{}个样本，占比{}".format(key, label_record.get(key), label_record.get(key) / df_length))


if __name__ == '__main__':
    train_df = pd.read_csv('/Users/chiang/数据/天池_零基础入门NLP_新闻文本分类/train_set.csv', sep='\t', nrows=100)
    print(train_df)
    print('-' * 30)

    # 本次赛题给定的文本比较长，每个句子平均由923个字符构成，最短的句子长度为64，最长的句子长度为7125。
    length_analysis(train_df)
    print('-' * 30)

    word_freq_analysis(train_df)
    print('-' * 30)

    label_freq_analysis(train_df)
