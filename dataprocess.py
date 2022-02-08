import jieba
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# -----读入csv文件、分词、设定好feature和label、划分训练集与测试集-----
# 这里的feature只用了“name”那一列

data = pd.read_csv('./data/POIs_dataset_test.csv')
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

# ----------
