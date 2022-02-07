# ---encoding:utf-8---
# @Time : 2021.02.19
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : test.py

import torch

# def repeat_by_dim(tensor, num=6, dim=1):
#     tensor = tensor.unsqueeze(dim)
#     tensor = [tensor] * num
#     tensor = torch.cat(tensor, dim=dim)
#     return tensor
#
#
# a = torch.range(1, 8).view([2, 4])
# print(a)
#
# c = repeat_by_dim(a, num=4, dim=1)
# print(c.size())
#
# c[0][0][0] = 100
# print(c)


# import csv
#
# with open('./data/transport_map.csv', 'r') as f:
#     reader = csv.reader(f)
#     result = list(reader)
#
# print('result\n', result[8-1][10-1])


import pickle

unique_user_list = pickle.load(open('./data/unique_user_list (1).pkl', 'rb'))
print('unique_user_list[{}]: {}'.format(len(unique_user_list), unique_user_list[:5]))

unique_pos_list = pickle.load(open('./data/unique_pos_list (1).pkl', 'rb'))
print('unique_pos_list[{}]: {}'.format(len(unique_pos_list), unique_pos_list[:5]))

print('unique_user_list[29379]',unique_user_list[29379])
print('unique_pos_list[1172]',unique_pos_list[1172])
print('unique_pos_list[1618]',unique_pos_list[1618])