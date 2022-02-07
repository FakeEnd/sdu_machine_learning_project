# ---encoding:utf-8---
# @Time : 2020.12.26
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : util_metric.py


'''
计算评价指标
'''

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import torch


# 计算分类指标
def caculate_metric(pred_y, labels, pred_prob, if_ROC=False):
    # print('labels', labels)
    # print('pred_y', pred_y)
    # print('pred_prob', pred_prob)

    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # print('tp\tfp\ttn\tfn')
    # print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

    ACC = float(tp + tn) / test_num

    # 精度 p
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # 敏感度 SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # 特异性 SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # 马修斯系数
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # AUC
    labels = labels.cpu()
    pred_prob = pred_prob.cpu()
    labels = labels.numpy().tolist()
    pred_prob = pred_prob.numpy().tolist()
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 默认1就是阳性
    AUC = auc(fpr, tpr)
    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])

    if if_ROC:
        # ROC(fpr, tpr, AUC)
        roc_data = [fpr, tpr, AUC]
        return metric, roc_data
    else:
        return metric

# 绘制ROC曲线
def ROC(fpr, tpr, roc_auc):
    # print('fpr, tpr, roc_auc')
    # print([fpr, tpr, roc_auc])

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.ylabel('True Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.title('Receiver operating characteristic example', fontdict={'weight': 'normal', 'size': 30})
    plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 30})
    plt.show()
