import pickle
import torch
import torch.utils.data as Data

import numpy as np
from util import util_data

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class DataManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.config = learner.config

        self.mode = self.config.mode

        # label: [15 11 10 ... 22  8  4]
        self.train_label = None
        self.test_label = None
        # raw_data_name: ['阿根廷共和国大使馆' '公共厕所' '中青旅遨游网(右安门店)']
        self.feature_name_train = None
        self.feature_name_test = None
        # raw_data_name: ['莲花池西路北京市海淀区培英小学南侧约150米' '东棉花胡同39号']
        self.feature_address_train = None
        self.feature_address_test = None
        # iterator
        self.train_dataloader = None
        self.test_dataloader = None

    def load_data(self):
        self.feature_name_train, self.feature_name_test, self.feature_address_train, self.feature_address_test, self.train_label, self.test_label = util_data.dataset_create()

        self.train_dataloader = self.construct_dataset(self.feature_name_train, self.feature_address_train, self.train_label, self.config.cuda,
                                                       self.config.batch_size)
        self.test_dataloader = self.construct_dataset(self.feature_name_test, self.feature_address_test, self.test_label, self.config.cuda,
                                                      self.config.batch_size)

    def construct_dataset(self, names, addresses, labels, cuda, batch_size):
        if cuda:
            labels = torch.cuda.LongTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        dataset = MyDataSet(names, addresses, labels)
        data_loader = Data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      drop_last=False,
                                      shuffle=True)
        # print('len(data_loader)', len(data_loader))
        return data_loader

    def get_dataloder(self, name):
        return_data = None
        if name == 'train_set':
            return_data = self.train_dataloader
        elif name == 'test_set':
            return_data = self.test_dataloader

        return return_data


class MyDataSet(Data.Dataset):
    def __init__(self, name, address, label):
        self.name = name
        self.address = address
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.name[idx], self.address[idx], self.label[idx]
