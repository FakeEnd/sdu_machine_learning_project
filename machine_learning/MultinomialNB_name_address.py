import jieba
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, roc_curve, auc, average_precision_score, precision_recall_curve, \
    precision_score, recall_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import DataLoader, TensorDataset

# 这里是使用了“name”和“address”两个特征，模型使用了朴素贝叶斯

data = pd.read_csv('../data/POIs_dataset_test.csv')
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

clf = MultinomialNB()
clf.fit(feature_train, labels_train)

y_pred = clf.predict(feature_test)
y_true = labels_test

# ROC画图
# fpr, tpr, threshold = roc_curve(y_true, y_pred)  ###计算真正率和假正率
# roc_auc = auc(fpr, tpr)
# print(roc_auc)
# PRC画图
# precision, recall, thresholds = precision_recall_curve(y_true, y_pred, pos_label=1)
# AP = average_precision_score(y_true, y_pred, average='macro', pos_label=1, sample_weight=None)

# [fpr, tpr, roc_auc], [recall, precision, AP]

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

