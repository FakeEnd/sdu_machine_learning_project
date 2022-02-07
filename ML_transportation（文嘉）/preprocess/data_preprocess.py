# ---encoding:utf-8---
# @Time : 2021.02.19
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : data_preprocess.py

import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data

from random import shuffle


def draw_hist(data):
    # 1）准备数据
    time = data

    # 2）创建画布
    plt.figure(figsize=(20, 8), dpi=100)

    # 3）绘制直方图
    # 设置组距
    distance = 2
    # 计算组数
    group_num = int((max(time) - min(time)) / distance)
    # 绘制直方图
    plt.hist(time, bins=group_num)

    # 修改x轴刻度显示
    plt.xticks(range(min(time), max(time))[::2])

    # 添加网格显示
    plt.grid(linestyle="--", alpha=0.5)

    # 添加x, y轴描述信息
    plt.xlabel("x")
    plt.ylabel("y")

    # 4）显示图像
    plt.show()


def check_data(data, len_threshold):
    len_list = []
    max_len = 0

    for i, traj in enumerate(data):
        cur_len = len(traj)
        if cur_len > len_threshold:
            print('[{}]: {}'.format(i, cur_len))

        len_list.append(cur_len)
        if cur_len > max_len:
            max_len = cur_len

    print('max_len', max_len)
    draw_hist(len_list)


def check_index(config, user_list, traj_pos_list, traj_time_list, label_pos_list, label_time_list):
    user = torch.max(user_list)
    if user >= config.user_vocab_size:
        print('ERROR USER')
    print('=' * 20, 'user_list check over', '=' * 20)

    pos = torch.max(traj_pos_list)
    if pos >= config.pos_vocab_size:
        print('ERROR TRAJ POS')
    print('=' * 20, 'traj_pos_list check over', '=' * 20)

    time = torch.max(traj_time_list)
    if time >= config.time_vocab_size:
        print('ERROR TRAJ TIME')
    print('=' * 20, 'traj_time_list check over', '=' * 20)

    pos = torch.max(label_pos_list)
    if pos >= config.pos_vocab_size:
        print('ERROR LABEL POS')
    print('=' * 20, 'label_pos_list check over', '=' * 20)

    time = torch.max(label_time_list)
    if time >= config.time_vocab_size:
        print('ERROR LABEL TIME')
    print('=' * 20, 'label_time_list check over', '=' * 20)


def construct_dataset(data, config):
    print('=' * 20, 'construct_dataset', '=' * 20)
    user_list = []
    traj_pos_list = []
    traj_time_list = []
    label_pos_list = []
    label_time_list = []

    for sample in data:
        user_id = sample[0]
        traj_pos = [traj[0] for traj in sample[2]]
        traj_time = [traj[1] for traj in sample[2]]
        label_pos = sample[1][0]
        label_time = sample[1][1]

        user_list.append(user_id)
        traj_pos_list.append(traj_pos)
        traj_time_list.append(traj_time)
        label_pos_list.append(label_pos)
        label_time_list.append(label_time)

    print('len(user_list)', len(user_list), user_list[:5])
    print('len(traj_pos_list)', len(traj_pos_list), traj_pos_list[:5])
    print('len(traj_time_list)', len(traj_time_list), traj_time_list[:5])
    print('len(label_pos_list)', len(label_pos_list), label_pos_list[:5])
    print('len(label_time_list)', len(label_time_list), label_time_list[:5])

    cuda = config.cuda
    batch_size = config.batch_size

    for i, user_id in enumerate(user_list):
        if user_id is None:
            print('error user_id[{}]'.format(i))
            user_list[i] = 39075  # len(unique_user_list) = 39075
        if label_pos_list[i] is None:
            print('error label_pos[{}]'.format(i))
            label_pos_list[i] = -4
        for j, pos in enumerate(traj_pos_list[i]):
            if pos is None:
                print('error traj_pos_list[{}][{}]'.format(i, j))
                traj_pos_list[i][j] = -4

    if cuda:
        user_list, traj_pos_list, traj_time_list, label_pos_list, label_time_list = torch.cuda.LongTensor(
            user_list), torch.cuda.LongTensor(traj_pos_list), torch.cuda.LongTensor(
            traj_time_list), torch.cuda.LongTensor(label_pos_list), torch.cuda.LongTensor(label_time_list)
    else:
        user_list, traj_pos_list, traj_time_list, label_pos_list, label_time_list = torch.LongTensor(
            user_list), torch.LongTensor(traj_pos_list), torch.LongTensor(
            traj_time_list), torch.LongTensor(label_pos_list), torch.LongTensor(label_time_list)

    print('-' * 20, '[construct_dataset]: check GPU data', '-' * 20)
    print('user_list.device:', user_list.device)

    print('-' * 20, '[construct_dataset]: check data shape', '-' * 20)
    print('user_list:', user_list.shape)  # [num_train_samples]
    print('traj_pos_list:', traj_pos_list.shape)  # [num_train_samples, len_threshold]
    print('traj_time_list:', traj_time_list.shape)  # [num_train_samples, len_threshold]
    print('label_pos_list:', label_pos_list.shape)  # [num_train_samples]
    print('label_time_list:', label_time_list.shape)  # [num_train_samples]

    # 对所有的索引+4,因为0-3已经被特殊符号占有
    # add_num = torch.ones_like(user_list, dtype=torch.int)
    # user_list = user_list + add_num
    add_num = torch.ones_like(traj_pos_list, dtype=torch.int) * 4
    traj_pos_list = traj_pos_list + add_num
    traj_time_list = traj_time_list + add_num
    add_num = torch.ones_like(label_pos_list, dtype=torch.int) * 4
    label_pos_list = label_pos_list + add_num
    label_time_list = label_time_list + add_num

    # 检查
    # check_index(configur, user_list, traj_pos_list, traj_time_list, label_pos_list, label_time_list)

    # TODO 考虑清楚输入的形式，这里输入应该要合并

    # batch_size:每次读出的一个batch有多大，shuffle:是否将DataSet中的数据随机打乱顺序
    data_loader = Data.DataLoader(MyDataSet(user_list, traj_pos_list, traj_time_list, label_pos_list, label_time_list),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)

    print('len(data_loader)', len(data_loader))
    print()
    return data_loader


# 继承pytorch.Data.Dataset类，用于封装训练数据，便于用Dataloader加载一个batch的数据
class MyDataSet(Data.Dataset):
    def __init__(self, user_list, traj_pos_list, traj_time_list, label_pos_list, label_time_list):
        self.user_list = user_list
        self.traj_pos_list = traj_pos_list
        self.traj_time_list = traj_time_list
        self.label_pos_list = label_pos_list
        self.label_time_list = label_time_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        return self.user_list[idx], self.traj_pos_list[idx], self.traj_time_list[idx], \
               self.label_pos_list[idx], self.label_time_list[idx]


def get_data(origin_data, config):
    len_threshold = 40
    '''
    大部分的轨迹长度都小于等于40
    '''
    # check_data(origin_data, len_threshold)

    selected_data = []
    token2idx = {'[PAD]': -1, '[CLS]': -2, '[SEP]': -3, '[MASK]': -4}

    # 根据轨迹长度阈值筛选数据
    for i, traj in enumerate(origin_data):
        traj_len = len(traj) - 1
        if traj_len <= len_threshold:
            user_id = traj[0]
            trajectory = [[-2, -2]] + traj[1:-1] + [[-3, -3]]
            last_pos = traj[-1]
            num_pad = len_threshold - traj_len

            trajectory.extend([[-1, -1]] * num_pad)

            sample = [user_id, last_pos, trajectory]
            selected_data.append(sample)

    print('len(selected_data)', len(selected_data), selected_data[:5])

    # 划分数据
    shuffle(selected_data)
    print('len(selected_data)', len(selected_data), selected_data[:5])
    print()

    train_data = selected_data[:int(0.8 * len(selected_data))]
    test_data = selected_data[int(0.8 * len(selected_data)):]

    data_loader_train = construct_dataset(train_data, config)
    data_loader_test = construct_dataset(test_data, config)

    return data_loader_train, data_loader_test


def get_data_train_test(train_data, test_data, config):
    len_threshold = 20
    '''
    大部分的轨迹长度都小于等于20
    '''
    # check_data(origin_data, len_threshold)

    selected_train_data = []
    selected_test_data = []
    token2idx = {'[PAD]': -1, '[CLS]': -2, '[SEP]': -3, '[MASK]': -4}

    # 根据轨迹长度阈值筛选数据
    for i, traj in enumerate(train_data):
        traj_len = len(traj) - 1
        if traj_len <= len_threshold:
            user_id = traj[0]
            trajectory = [[-2, -2]] + traj[1:-1] + [[-3, -3]]
            last_pos = traj[-1]
            num_pad = len_threshold - traj_len
            trajectory.extend([[-1, -1]] * num_pad)

            sample = [user_id, last_pos, trajectory]
            selected_train_data.append(sample)

            '''
            数据增强
            '''
            k_slice = 20
            for k in range(1, k_slice):
                # k->1
                if traj_len >= k + 2:
                    trajectory = [[-2, -2]] + traj[1:1 + k] + [[-3, -3]]
                    last_pos = traj[1 + k]
                    num_pad = len_threshold - len(trajectory) + 2 - 1
                    trajectory.extend([[-1, -1]] * num_pad)
                    sample = [user_id, last_pos, trajectory]
                    selected_train_data.append(sample)

            # # 1->1
            # if traj_len >= 3:
            #     trajectory = [[-2, -2]] + traj[1:1 + 1] + [[-3, -3]]
            #     last_pos = traj[1 + 1]
            #     num_pad = len_threshold - len(trajectory) + 2 - 1
            #     trajectory.extend([[-1, -1]] * num_pad)
            #     sample = [user_id, last_pos, trajectory]
            #     selected_train_data.append(sample)
            #
            # # 2->1
            # if traj_len >= 4:
            #     trajectory = [[-2, -2]] + traj[1:1 + 2] + [[-3, -3]]
            #     last_pos = traj[1 + 2]
            #     num_pad = len_threshold - len(trajectory) + 2 - 1
            #     trajectory.extend([[-1, -1]] * num_pad)
            #     sample = [user_id, last_pos, trajectory]
            #     selected_train_data.append(sample)
            #
            # # 3->1
            # if traj_len >= 5:
            #     trajectory = [[-2, -2]] + traj[1:1 + 3] + [[-3, -3]]
            #     last_pos = traj[1 + 3]
            #     num_pad = len_threshold - len(trajectory) + 2 - 1
            #     trajectory.extend([[-1, -1]] * num_pad)
            #     sample = [user_id, last_pos, trajectory]
            #     selected_train_data.append(sample)
            #
            # # 5->1
            # if traj_len >= 7:
            #     trajectory = [[-2, -2]] + traj[1:1 + 4] + [[-3, -3]]
            #     last_pos = traj[1 + 4]
            #     num_pad = len_threshold - len(trajectory) + 2 - 1
            #     trajectory.extend([[-1, -1]] * num_pad)
            #     sample = [user_id, last_pos, trajectory]
            #     selected_train_data.append(sample)

        else:
            user_id = traj[0]
            trajectory = [[-2, -2]] + traj[-len_threshold:-1] + [[-3, -3]]
            last_pos = traj[-1]
            num_pad = len_threshold - len(trajectory) + 2 - 1
            trajectory.extend([[-1, -1]] * num_pad)

            sample = [user_id, last_pos, trajectory]
            selected_train_data.append(sample)

    print('len(selected_train_data)', len(selected_train_data), selected_train_data[:5])

    for i, traj in enumerate(test_data):
        traj_len = len(traj) - 1
        if traj_len <= len_threshold:
            traj_len = len(traj) - 1
            user_id = traj[0]
            trajectory = [[-2, -2]] + traj[1:-1] + [[-3, -3]]
            last_pos = traj[-1]
            num_pad = len_threshold - traj_len

            trajectory.extend([[-1, -1]] * num_pad)

            sample = [user_id, last_pos, trajectory]
            selected_test_data.append(sample)

        else:
            user_id = traj[0]
            trajectory = [[-2, -2]] + traj[-len_threshold:-1] + [[-3, -3]]
            last_pos = traj[-1]
            num_pad = len_threshold - len(trajectory) + 2 - 1
            trajectory.extend([[-1, -1]] * num_pad)

            sample = [user_id, last_pos, trajectory]
            selected_test_data.append(sample)

    print('len(selected_test_data)', len(selected_test_data), selected_test_data[:5])

    # 划分数据
    shuffle(selected_train_data)
    shuffle(selected_test_data)
    print('len(selected_train_data)', len(selected_train_data), selected_train_data[:5])
    print('len(selected_test_data)', len(selected_test_data), selected_test_data[:5])
    print()

    data_loader_test = construct_dataset(selected_test_data, config)
    data_loader_train = construct_dataset(selected_train_data, config)

    return data_loader_train, data_loader_test


def get_data_final_test(test_data, config):
    len_threshold = 20
    selected_test_data = []
    # check_data(test_data, len_threshold)
    for i, traj in enumerate(test_data):
        traj_len = len(traj) - 1
        if traj_len <= len_threshold:
            traj_len = len(traj) - 1
            user_id = traj[0]
            trajectory = [[-2, -2]] + traj[1:-1] + [[-3, -3]]
            last_pos = traj[-1]
            num_pad = len_threshold - traj_len

            trajectory.extend([[-1, -1]] * num_pad)

            sample = [user_id, last_pos, trajectory]
            selected_test_data.append(sample)

        else:
            user_id = traj[0]
            trajectory = [[-2, -2]] + traj[-len_threshold + 1:] + [[-3, -3]]
            last_pos = traj[-1]
            num_pad = len_threshold - len(trajectory) + 2 - 1
            trajectory.extend([[-1, -1]] * num_pad)

            sample = [user_id, last_pos, trajectory]
            selected_test_data.append(sample)

    print('len(selected_test_data)', len(selected_test_data), selected_test_data[:5])
    data_loader_test = construct_dataset(selected_test_data, config)
    return data_loader_test


if __name__ == '__main__':
    import pickle
    from configur import config

    test_data = pickle.load(open('../data/test_embed_data (1).pkl', 'rb'))
    print('test_data[{}]: {}'.format(len(test_data), test_data[:5]))

    config_train = config.get_train_config()
    config_train.user_vocab_size = 39075 + 1
    config_train.time_vocab_size = 1440 + 4
    config_train.pos_vocab_size = 1902 + 4
    config_train.max_len = 51

    test_data = get_data_final_test(test_data, config_train)
