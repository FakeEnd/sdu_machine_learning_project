#coding:utf-8
import jieba
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 划分训练集和测试集
def dataset_create(filename= '/mnt/sdb/home/jjr/sdu-machine_learning/data/POIs_dataset_test.csv'):

    data = pd.read_csv(filename)
    feature_name = data['name'].values
    feature_address = data['address'].values
    labels = data['type'].values
    labels = [ x - 1 for x in labels]

    feature_name_train, feature_name_test, feature_address_train, feature_address_test, labels_train, labels_test = train_test_split(
        feature_name, feature_address, labels, test_size=0.15, random_state=66, stratify=labels)
    return feature_name_train, feature_name_test, feature_address_train, feature_address_test, labels_train, labels_test


if __name__ == '__main__':
    feature_name_train, feature_name_test, feature_address_train, feature_address_test, labels_train, labels_test= dataset_create()
    print(feature_name_train, feature_name_test, feature_address_train, feature_address_test, labels_train, labels_test)