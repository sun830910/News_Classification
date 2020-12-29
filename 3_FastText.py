# -*- coding: utf-8 -*-

"""
Created on 12/29/20 7:04 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import pandas as pd
from sklearn.metrics import f1_score
import fasttext

if __name__ == '__main__':
    # 转换为FastText需要的格式
    train_df = pd.read_csv('/Users/chiang/数据/天池_零基础入门NLP_新闻文本分类/train_set.csv', sep='\t', nrows=15000)
    train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
    train_df[['text', 'label_ft']].iloc[:-5000].to_csv('FastText_train.csv', index=None, header=None, sep='\t')
    print(train_df)

    model = fasttext.train_supervised('FastText_train.csv', lr=1, wordNgrams=3, verbose=2, minCount=1, epoch=50,
                                      loss='hs')
    val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
    print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
