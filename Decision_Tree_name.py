import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pandas import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score,cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm,tree
from torch.utils.data import DataLoader, TensorDataset


data = pd.read_csv('./data/POIs_dataset_test.csv')
feature_name = data['name'].values
feature_address = data['address'].values
labels = data['type'].values

tokenized_list1 = []
for text in feature_name:
    words = [word for word in jieba.cut(text)]
    tokenized_list1.append(' '.join(words))

counter1 = CountVectorizer()
counts1 = counter1.fit_transform(tokenized_list1)
feature_name = torch.Tensor(counts1.toarray())

tokenized_list2 = []
for text in feature_address:
    words = [word for word in jieba.cut(text)]
    tokenized_list2.append(' '.join(words))

counter2 = CountVectorizer()
counts2 = counter2.fit_transform(tokenized_list1)
feature_address = torch.Tensor(counts2.toarray())

# 按dim=1才是每个向量的首尾拼接
feature = torch.cat((feature_name, feature_address), dim=1)

feature_train, feature_test, labels_train, labels_test = train_test_split(
    feature, labels, test_size=0.15, random_state=66, stratify=labels)

clf = tree.DecisionTreeClassifier()
clf.fit(feature_train,labels_train)
y_pred = clf.predict(feature_test)
y_true = labels_test
PrecisionScore = precision_score(y_true, y_pred, average='macro')
AccuracyScore = accuracy_score(y_true, y_pred)
RecallScore = recall_score(y_true, y_pred, average='macro')
F1Score = 2*RecallScore*PrecisionScore/(PrecisionScore+RecallScore)
KappaScore = cohen_kappa_score(y_true, y_pred)
print('精确率：', PrecisionScore)
print('准确率：', AccuracyScore)
print('召回率：', RecallScore)
print('F1-score:', F1Score)
print('KappaScore:', KappaScore)