# ---encoding:utf-8---
# @Time : 2021.02.17
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : main.py


import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import time

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from configur import config
from util import top1_accuracy, top5_accuracy
from model import BERT
from preprocess import data_preprocess


# 用训练集进行训练
def train_process(train_iter, valid_iter, test_iter, iter_k, model, optimizer, criterion, config):
    steps = 0  # 1个batch对应1个step
    best_acc = 0
    best_performance = 0

    # 每一次交叉验证都迭代config.epoch个epoch
    for epoch in range(1, config.epoch + 1):
        # 遍历整个训练集
        for batch in train_iter:
            user, input_pos, input_time, label_pos, label_time = batch
            logits, output = model(user, input_pos, input_time)

            for pos in label_pos:
                if pos >= config.num_class:
                    print('error pos', pos)

            loss = criterion(logits, label_pos)
            # loss = criterion(logits.view(-1, configur.num_class), label_pos.view(-1))
            loss = (loss.float()).mean()

            # flooding method
            b = 0.06
            loss = (loss - b).abs() + b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

            # 每训练config.log_interval个batch/step就打印一次训练结果，并记录训练集当前的损失和准确度，用于绘图
            if steps % config.interval_log == 0:
                # torch.max(logits, 1)函数：返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                corrects = (torch.max(logits, 1)[1] == label_pos).sum()  # .view(label.size()此处没有影响
                # corrects += (torch.max(logits, 1)[1].view(label.size()) == label).sum()

                # 因为不是每个batch的大小都等于cofig.batch_size
                the_batch_size = label_pos.shape[0]
                train_acc = 100.0 * corrects / the_batch_size

                top1_acc = top1_accuracy.top1_accuracy(logits, label_pos) / label_pos.shape[0]
                top5_acc = top5_accuracy.top5_accuracy(logits, label_pos) / label_pos.shape[0]

                sys.stdout.write(
                    '\rEpoch[{}] Batch[{}] - loss: {:.6f}  ACC: {:.4f}%({}/{})  '
                    'top1_acc: {:.4f}  top5_acc: {:.4f}'.format(
                        epoch, steps,
                        loss,
                        train_acc,
                        corrects,
                        the_batch_size,
                        top1_acc,
                        top5_acc
                    ))
                print()

                # 绘图数据
                step_log_interval.append(steps)
                train_acc_record.append(train_acc)  # 训练集的准确度
                train_loss_record.append(loss)  # 训练集的平均损失

        sum_epoch = iter_k * config.epoch + epoch

        # 每config.test_interval个epoch迭代完都测试一下在测试集上的表现，并记录当前测试集的损失和准确度，用于绘图
        if test_iter and sum_epoch % config.interval_test == 0:
            print('#' * 60 + 'Periodic Test' + '#' * 60)
            test_acc, test_loss, test_metric = model_eval(test_iter, model, config)
            print('test current performance')
            print('[top1 ACC\ttop5 ACC]')
            print(test_metric)
            print('#' * 60 + 'Over' + '#' * 60)

            # 绘图数据
            step_test_interval.append(sum_epoch)
            top1_acc_record.append(test_metric[0])  # 测试集的准确度
            top5_acc_record.append(test_metric[1])  # 测试集的损失

            # 满足一定的条件就保存当前的模型
            if test_acc > best_acc:
                best_acc = test_acc
                best_performance = test_metric
                if config.save_best and best_acc > config.threshold:
                    print('Save Model: {}, ACC: {:.4f}%\n'.format(config.learn_name, best_acc))
                    save_model(model.state_dict(), best_acc, config.path_model_save, config.learn_name)

        # 绘图
        show_res(config_train)

    return best_performance, best_acc


# 对模型进行测试
def model_eval(data_iter, model, config):
    iter_size, corrects, avg_loss = 0, 0, 0
    print('model_eval data_iter', len(data_iter))
    top1_acc = 0
    top5_acc = 0

    # 不写容易爆内存
    i = 0
    with torch.no_grad():
        for batch in data_iter:
            user, input_pos, input_time, label_pos, label_time = batch
            label = label_pos
            logits, output = model(user, input_pos, input_time)

            for pos in label_pos:
                if pos >= config.num_class:
                    print('error pos', pos)

            batch_top1 = top1_accuracy.top1_accuracy(logits, label_pos)
            batch_top5 = top5_accuracy.top5_accuracy(logits, label_pos)
            top1_acc += batch_top1
            top5_acc += batch_top5

            pred_prob_all = F.softmax(logits, dim=1)  # 预测概率 [batch_size, class_num]
            # pred_prob_positive = pred_prob_all[:, 1]  # 注意，极其容易出错
            pred_prob_sort = torch.max(pred_prob_all, 1)  # 每个样本中预测的最大的概率 [batch_size]
            pred_class = pred_prob_sort[1]  # 每个样本中预测的最大的概率所在的位置（类别） [batch_size]
            # corrects += (torch.max(logits, 1)[1] == label).sum()
            corrects += (pred_class == label).sum()

            iter_size += label.shape[0]

            batch_top1_acc = batch_top1 / label.shape[0]
            batch_top5_acc = batch_top5 / label.shape[0]
            print('Test[{}] -  batch_top1_acc: {:.4f}%  batch_top5_acc: {:.4f}%'.format(
                i, batch_top1_acc, batch_top5_acc))
            i += 1

    avg_loss /= iter_size
    accuracy = 100.0 * corrects / iter_size
    top1_acc = top1_acc / iter_size
    top5_acc = top5_acc / iter_size
    print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})  top1_acc: {:.4f}%  top5_acc: {:.4f}%'.format(
        avg_loss, accuracy, corrects, iter_size, top1_acc, top5_acc))
    metric = [top1_acc, top5_acc]
    return accuracy, avg_loss, metric


# 保存模型
def save_model(model_dict, best_acc, save_dir, save_prefix):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = 'ACC->{:.4f}, {}.pt'.format(best_acc, save_prefix)
    save_path_pt = os.path.join(save_dir, filename)
    torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)


# 绘图
def show_res(config_train):
    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Top1 Acc Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(step_test_interval, top1_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Top5 Acc Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(step_test_interval, top5_acc_record)

    # plt.savefig('../figure/' + config_train.learn_name + '.png')
    plt.show()


# 训练和测试
def train_test(train_iter_orgin, test_iter, config_train):
    print('=' * 50, 'train-test', '=' * 50)
    print('len(train_iter)', len(train_iter_orgin))  # 训练集的batch数
    print('len(test_iter)', len(test_iter))  # 测试集的batch数

    # 数据记录容器,用于绘图
    global step_log_interval, step_test_interval, \
        train_acc_record, train_loss_record, top1_acc_record, top5_acc_record

    step_log_interval = []
    step_test_interval = []

    train_acc_record = []
    train_loss_record = []
    top1_acc_record = []
    top5_acc_record = []

    # 创建模型
    model = BERT.BERT(config_train)

    if config_train.cuda: model.cuda()

    print('-' * 50, 'Model.named_parameters', '-' * 50)
    for name, value in model.named_parameters():
        print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))

    # 计算参数总量
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        # print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        # print("该层参数和：" + str(l))
        k = k + l
    print('=' * 50, "总参数数量:" + str(k), '=' * 50)

    # 选择优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=config_train.lr,
    #                              weight_decay=config_train.reg)  # L2正则化
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config_train.lr, weight_decay=config_train.reg)
    # optimizer = torch.optim.Adagrad(params=model.parameters(), lr=config_train.lr)
    # optimizer = torch.optim.Adamax(params=model.parameters(), lr=config_train.lr)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=config_train.lr)

    criterion = nn.CrossEntropyLoss()
    model.train()

    # 训练
    print('=' * 50 + 'Start Training' + '=' * 50)
    best_performance, best_acc = train_process(train_iter_orgin, None, test_iter, 0, model, optimizer, criterion,
                                               config_train)

    # 绘图
    show_res(config_train)

    # 测试集测试
    print('*' * 60 + 'The Last Test' + '*' * 60)
    test_acc, test_loss, test_metric = model_eval(test_iter, model, config_train)
    print('last test performance')
    print('\t[top1 ACC\ttop5 ACC]')
    print('\t{}'.format(test_metric))
    print()
    print('best_performance')
    print('\t[top1 ACC\ttop5 ACC]')
    print('\t{}'.format(best_performance))

    # 满足一定的条件就保存当前的模型
    if test_acc > best_acc:
        best_acc = test_acc
        if config_train.save_best and best_acc >= config_train.threshold:
            print('Save Model: {}, ACC: {:.4f}%\n'.format(config_train.learn_name, best_acc))
            save_model(model.state_dict(), best_acc, config_train.path_model_save, config_train.learn_name)

    print('*' * 60 + 'The Last Test Over' + '*' * 60)
    return model


if __name__ == '__main__':
    # 计时开始
    time_start = time.time()

    # 读取相关数据
    unique_user_list = pickle.load(open('../data/unique_user_list (1).pkl', 'rb'))
    print('unique_user_list[{}]: {}'.format(len(unique_user_list), unique_user_list[:5]))

    unique_pos_list = pickle.load(open('../data/unique_pos_list (1).pkl', 'rb'))
    print('unique_pos_list[{}]: {}'.format(len(unique_pos_list), unique_pos_list[:5]))

    origin_data = pickle.load(open('../data/embed_data (1).pkl', 'rb'))
    print('origin_data[{}]: {}'.format(len(origin_data), origin_data[:5]))

    test_data = pickle.load(open('../data/test_embed_data (1).pkl', 'rb'))
    print('test_data[{}]: {}'.format(len(test_data), test_data[:5]))

    road_map_embedding = np.load('../data/road_map_embedding (2).npy')
    print('road_map_embedding[{}]: {}'.format(len(road_map_embedding), road_map_embedding[:2]))

    transport_map_embedding = np.load('../data/transport_map_embedding (2).npy')
    print('transport_map_embedding[{}]: {}'.format(len(transport_map_embedding), transport_map_embedding[:2]))

    user_embedding = pickle.load(open('../data/user_embeding.pkl', 'rb'))
    print('user_embedding[{}]: {}'.format(len(user_embedding), user_embedding[:2]))

    # 设置打印格式
    np.set_printoptions(linewidth=400, precision=4)

    # 加载配置
    config_train = config.get_train_config()
    config_train.user_vocab_size = 39075 + 1
    config_train.time_vocab_size = 1440 + 4
    config_train.pos_vocab_size = 1902 + 4
    config_train.max_len = 51
    config_train.road_map_embedding = road_map_embedding
    config_train.transport_map_embedding = transport_map_embedding
    config_train.user_embedding = user_embedding

    # 选择GPU
    torch.cuda.set_device(config_train.device)

    # 加载数据
    # train_iter, test_iter = data_preprocess.get_data(origin_data, config_train)
    train_iter, test_iter = data_preprocess.get_data_train_test(origin_data, test_data, config_train)

    model = train_test(train_iter, test_iter, config_train)
    # model = train_test(test_iter, train_iter, config_train)

    save_model(model.state_dict(), 0000, config_train.path_model_save, config_train.learn_name)

    # 计时结束
    time_end = time.time()
    print('total time cost', time_end - time_start, 'seconds')
