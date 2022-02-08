import jieba
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import DataLoader, TensorDataset

# 这里是只使用了“name”这个特征，模型使用了朴素贝叶斯

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

clf = MultinomialNB()
clf.fit(feature_name_train,labels_train)

y_pred = clf.predict(feature_name_test)
y_true = labels_test

score = accuracy_score(y_true, y_pred)

print(score)

