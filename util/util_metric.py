import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
import numpy as np

def CAN(softmax_logits, label_real):
    y_pred = softmax_logits.cpu().numpy()
    num_classes = 2
    y_true = label_real.numpy()
    acc_original = np.mean([y_pred.argmax(1) == y_true])
    print('original acc: %s' % acc_original)

    # 从训练集统计先验分布
    # prior = np.zeros(num_classes)
    # for d in train_data:
    #     prior[d[1]] += 1.
    # prior /= prior.sum()
    prior = np.array([0.5, 0.5])

    # 评价每个预测结果的不确定性
    k = 2
    y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)  # 归一化
    y_pred_entropy = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)  # top-k熵
    # print(y_pred_entropy)

    # 选择阈值，划分高、低置信度两部分
    threshold = 0.95
    y_pred_confident = y_pred[y_pred_entropy < threshold]  # top-k熵低于阈值的是高置信度样本
    y_pred_unconfident = y_pred[y_pred_entropy >= threshold]  # top-k熵高于阈值的是低置信度样本
    y_true_confident = y_true[y_pred_entropy < threshold]
    y_true_unconfident = y_true[y_pred_entropy >= threshold]

    # 显示两部分各自的准确率
    # 一般而言，高置信度集准确率会远高于低置信度的
    num = (y_pred_confident.argmax(1) == y_true_confident)
    if len(num) == 0:
        acc_confident = 0
    else:
        acc_confident = num.mean()

    # acc_unconfident = (y_pred_unconfident.argmax(1) == y_true_unconfident).mean()
    # print('confident acc: %s' % acc_confident)
    # print('unconfident acc: %s' % acc_unconfident)

    # 逐个修改低置信度样本，并重新评价准确率
    right, alpha, iters = 0, 1, 1  # 正确的个数，alpha次方，iters迭代次数
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0)  # Y is L_0
        for _ in range(iters):
            Y = Y ** alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        if y.argmax() == y_true_unconfident[i]:
            right += 1

    # 输出修正后的准确率
    acc_final = (acc_confident * len(y_pred_confident) + right) / len(y_pred)

    print('*' * 50)
    print('acc_final',acc_final)
    print('*' * 50)
    return acc_final


def get_binary_classification_metrics(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # print('Sum[{}]: TP[{}], TN[{}], FP[{}], FN[{}]'.format(tp + tn + fp + fn, tp, tn, fp, fn))

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)  # Default 1 is positive sample
    AUC = auc(FPR, TPR)

    # PRC and AP
    # precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    # AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    # ROC(FPR, TPR, AUC)
    # PRC(Recall, Precision, AP)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = None
    prc_data = None
    # roc_data = [FPR, TPR, AUC]
    # prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data