import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pandas import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm, tree
from torch.utils.data import DataLoader, TensorDataset

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

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovo', gamma='scale', kernel='rbf',
              probability=False, random_state=None, shrinking=True,
              verbose=False)

clf.fit(feature_name_train, labels_train)
y_pred = clf.predict(feature_name_test)
y_true = labels_test
PrecisionScore = precision_score(y_true, y_pred, average='macro')
AccuracyScore = accuracy_score(y_true, y_pred)
RecallScore = recall_score(y_true, y_pred, average='macro')
F1Score = 2 * RecallScore * PrecisionScore / (PrecisionScore + RecallScore)
KappaScore = cohen_kappa_score(y_true, y_pred)
print('精确率：', PrecisionScore)
print('准确率：', AccuracyScore)
print('召回率：', RecallScore)
print('F1-score:', F1Score)
print('KappaScore:', KappaScore)
