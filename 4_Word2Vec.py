# -*- coding: utf-8 -*-

"""
Created on 12/29/20 8:16 PM
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import logging
import random
import numpy as np
import torch
import pandas as pd
from gensim.models.word2vec import Word2Vec

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set seed
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# split data to 10 fold
fold_num = 10
data_file = '/Users/chiang/数据/天池_零基础入门NLP_新闻文本分类/train_set.csv'


def all_data2fold(fold_num, num=10000):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()[:num]  # df to list
    labels = f['label'].tolist()[:num]

    total = len(labels)  # num of samples

    index = list(range(total))  # list of index
    np.random.shuffle(index)

    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)

        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))

    return fold_data


def train_embedding():
    fold_data = all_data2fold(10)
    # build train data for word2vec
    fold_id = 9

    train_texts = []
    for i in range(0, fold_id):
        data = fold_data[i]
        train_texts.extend(data['text'])

    logging.info('Total %d docs.' % len(train_texts))
    logging.info('Start training...')
    num_features = 100  # Word vector dimensionality
    num_workers = 8  # Number of threads to run in parallel

    train_texts = list(map(lambda x: list(x.split()), train_texts))
    model = Word2Vec(train_texts, workers=num_workers, size=num_features)
    model.init_sims(replace=True)

    # save model
    model.save("./Embedding/word2vec.bin")
    model.wv.save_word2vec_format('./Embedding/word2vec.txt', binary=False)


if __name__ == '__main__':
    train_embedding()
