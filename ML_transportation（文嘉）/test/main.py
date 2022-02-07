# ---encoding:utf-8---
# @Time : 2021.02.17
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : main.py

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configur import config
from preprocess import data_preprocess
from model import BERT
from util import predict_pos


def load_model(new_model, path_trained_model):
    trained_dict = torch.load(path_trained_model)
    # print('trained_dict', trained_dict)

    # 获取新模型的参数
    new_model_dict = new_model.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
    trained_dict = {k: v for k, v in trained_dict.items() if k in new_model_dict}
    # 更新现有的model_dict
    new_model_dict.update(trained_dict)
    # 加载我们真正需要的state_dict
    new_model.load_state_dict(new_model_dict)

    return new_model


# 对模型进行测试
def model_test(data_iter, model, config):
    iter_size, corrects, avg_loss = 0, 0, 0
    print('model_eval data_iter', len(data_iter))
    top1_acc = 0
    top5_acc = 0

    # 不写容易爆内存
    i = 0
    with torch.no_grad():
        for batch in data_iter:
            user, input_pos, input_time, label_pos, label_time = batch
            logits, output = model(user, input_pos, input_time)

            for pos in label_pos:
                if pos >= config.num_class:
                    print('error pos', pos)

            predict_pos.predict_pos(logits)
            print('Test[{}]'.format(i))
            i += 1


if __name__ == '__main__':
    road_map_embedding = np.load('../data/road_map_embedding (2).npy')
    print('road_map_embedding[{}]: {}'.format(len(road_map_embedding), road_map_embedding[:2]))

    transport_map_embedding = np.load('../data/transport_map_embedding (2).npy')
    print('transport_map_embedding[{}]: {}'.format(len(transport_map_embedding), transport_map_embedding[:2]))

    user_embedding = pickle.load(open('../data/user_embeding.pkl', 'rb'))
    print('user_embedding[{}]: {}'.format(len(user_embedding), user_embedding[:2]))

    test_data = pickle.load(open('../data/test_embed_data (1).pkl', 'rb'))
    print('test_data[{}]: {}'.format(len(test_data), test_data[:5]))

    # 设置打印格式
    np.set_printoptions(linewidth=400, precision=4)

    config_train = config.get_train_config()
    config_train.user_vocab_size = 39075 + 1
    config_train.time_vocab_size = 1440 + 4
    config_train.pos_vocab_size = 1902 + 4
    config_train.max_len = 51
    config_train.road_map_embedding = road_map_embedding
    config_train.transport_map_embedding = transport_map_embedding
    config_train.user_embedding = user_embedding
    config_train.path_trained_model = '../model_save/common_train/ACC->38.7990, train_22.pt'

    # 选择GPU
    torch.cuda.set_device(config_train.device)

    test_data = data_preprocess.get_data_final_test(test_data, config_train)

    model = BERT.BERT(config_train)
    if config_train.cuda: model.cuda()

    model = load_model(model, config_train.path_trained_model)

    model_test(test_data, model, config_train)
