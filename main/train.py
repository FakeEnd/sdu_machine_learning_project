import sys
import os
import pickle

import torch

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from configuration import config_init
from frame import Learner


def SL_train(config):
    torch.cuda.set_device(config.device)
    roc_datas, prc_datas = [], []

    # ToDo 两种的kmers的更改
    if config.model == 'FusionDNAbert':
        config.kmers = [3, 6]

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.adjust_model()
    # learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()

    roc_datas.append(learner.visualizer.roc_data)
    prc_datas.append(learner.visualizer.prc_data)


def SL_fintune():
    # config = config_SL.get_config()
    config = pickle.load(open('../result/jobID/config.pkl', 'rb'))
    config.path_params = '../result/jobID/DNAbert, MCC[0.64].pt'
    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICE'] = '1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    config = config_init.get_config()
    SL_train(config)
