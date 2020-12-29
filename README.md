# 天池 零基础入门NLP - 新闻文本分类

# 简介

## 赛题名称

零基础入门NLP - 新闻文本分类

## 赛题目标

对新闻文本进行分类，一共14类。

## 赛题数据

新闻文本，并按照字符级别进行匿名处理，共有14类，包含财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐的文本数据。

训练集20w条样本，测试集A包括5w条样本，测试集B包括5w条样本。

处理后的赛题训练数据如下：

![Image](http://jupter-oss.oss-cn-hangzhou.aliyuncs.com/public/files/image/1095279501877/1594906820936_hVKPJHWvu4.jpg)

在数据集中标签的对应的关系如下：{'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}

## 评测指标

评价标准为类别`f1_score`的均值，选手提交结果与实际测试集的类别进行对比，结果越大越好。



# 数据分析

## 代码路径

0_Data_Analysis.py

## 介绍

抽取训练集中100个样本进行数据分析，得到以下结论。

## 数据结论

1. 本次赛题给定的文本比较长，每个句子平均由923个字符构成，最短的句子长度为64，最长的句子长度为7125。

2. 在训练集中总共包括2405个字，其中编号3750的字出现的次数最多，编号5034的字出现的次数最少。

3. 各类标签有不平衡的问题，其中最多的是科技类新闻，最少的是星座类新闻。

4. 样本偏长，需要做截断。
5. 类别不均衡会影响模型的精度。

# 方法

## 1. 词袋模型 + 线性模型

### 代码路径

1_Count_Vectors_RidgeClassifier.py

### 实验结果

F1_Score = 0.7415

## 2. TF-IDF + 线性模型

### 代码路径

2_TFIDF_RidgeClassifier.py

### 实验结果

F1_Score = 0.8722

## 3. FastText

### 代码路径

3_FastText.py

### 实验结果

lr=0.1、wordNgrams=3、epoch=50时，F1_Score=0.7298

lr=1、wordNgrams=3、epoch=50时，F1_Score=0.8275