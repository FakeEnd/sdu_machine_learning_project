#coding:utf-8
import jieba
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# import os
# homedir = os.getcwd()
# print(homedir)

# 划分训练集和测试集
def dataset_create(filename= '/mnt/sdb/home/jjr/sdu-machine_learning/data/POIs_dataset_test.csv'):

    data = pd.read_csv(filename)
    feature_name = data['name'].values
    labels = data['type'].values

    tokenized_list = []
    for text in feature_name:
        words = [word for word in jieba.cut(text)]
        tokenized_list.append(' '.join(words))

    counter = CountVectorizer()
    counts = counter.fit_transform(tokenized_list)
    feature_name = torch.Tensor(counts.toarray())

    feature_name_train, feature_name_test, labels_train, labels_test = train_test_split(
        feature_name, labels, test_size=0.15, random_state=66, stratify=labels)
    return feature_name_train, feature_name_test, labels_train, labels_test


if __name__ == '__main__':
    feature_name_train, feature_name_test, labels_train, labels_test = dataset_create()
    print(feature_name_train, feature_name_test, labels_train, labels_test)