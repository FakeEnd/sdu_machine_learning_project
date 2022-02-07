# ---encoding:utf-8---
# @Time : 2020.10.30
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : config_data.py

import argparse


def get_train_config():
    parse = argparse.ArgumentParser(description='common train configur')

    # 项目配置参数
    parse.add_argument('-learn-name', type=str, default='train_55', help='本次训练的名称')
    parse.add_argument('-path-model-save', type=str, default='../model_save/common_train/', help='保存训练模型位置')
    parse.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
    parse.add_argument('-threshold', type=float, default=0, help='准确率阈值，单位: %')

    # 训练参数
    # parse.add_argument('-lr', type=float, default=0.0001, help='学习率')
    parse.add_argument('-k-fold', type=int, default=-1, help='k折交叉验证,-1代表只使用train-test方式')
    parse.add_argument('-num-class', type=int, default=1902 + 4, help='类别数量')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=1)
    parse.add_argument('-interval-log', type=int, default=50, help='经过多少batch记录一次训练状态')
    parse.add_argument('-interval-test', type=int, default=5, help='经过多少epoch对测试集进行测试')
    parse.add_argument('-static', type=bool, default=False, help='嵌入是否冻结')

    # # 模型参数
    # parse.add_argument('-lr', type=float, default=0.0002, help='学习率')
    # parse.add_argument('-reg', type=float, default=0.0001, help='正则化lambda')
    # parse.add_argument('-batch-size', type=int, default=2048 * 2, help='一个batch中有多少个sample')
    # parse.add_argument('-epoch', type=int, default=50, help='迭代次数')
    # parse.add_argument('-dim-embedding', type=int, default=32 * 2, help='词（残基）向量的嵌入维度')
    # parse.add_argument('-num-layer', type=int, default=8, help='Transformer的Encoder模块的堆叠层数')
    # parse.add_argument('-num-head', type=int, default=8, help='多头注意力机制的头数')
    # parse.add_argument('-dim-feedforward', type=int, default=32 * 6, help='词（残基）向量的嵌入维度')
    # parse.add_argument('-dim-k', type=int, default=32 * 2, help='k/q向量的嵌入维度')
    # parse.add_argument('-dim-v', type=int, default=32 * 2, help='v向量的嵌入维度')

    # com-embed模型参数
    # 目前最佳配置,测试top1可达39.50%,证明参数不必太大
    # parse.add_argument('-lr', type=float, default=0.0002, help='学习率')
    # parse.add_argument('-reg', type=float, default=0.0001, help='正则化lambda')
    # parse.add_argument('-batch-size', type=int, default=1024 * 2, help='一个batch中有多少个sample')
    # parse.add_argument('-epoch', type=int, default=50, help='迭代次数')
    # parse.add_argument('-dim-embedding', type=int, default=32, help='词（残基）向量的嵌入维度')
    # parse.add_argument('-num-layer', type=int, default=4, help='Transformer的Encoder模块的堆叠层数')
    # parse.add_argument('-num-head', type=int, default=12, help='多头注意力机制的头数')
    # parse.add_argument('-dim-feedforward', type=int, default=64, help='词（残基）向量的嵌入维度')
    # parse.add_argument('-dim-k', type=int, default=32, help='k/q向量的嵌入维度')
    # parse.add_argument('-dim-v', type=int, default=32, help='v向量的嵌入维度')

    parse.add_argument('-lr', type=float, default=0.0002, help='学习率')
    parse.add_argument('-reg', type=float, default=0.0001, help='正则化lambda')
    parse.add_argument('-batch-size', type=int, default=1024 * 2, help='一个batch中有多少个sample')
    parse.add_argument('-epoch', type=int, default=50, help='迭代次数')
    parse.add_argument('-dim-embedding', type=int, default=8, help='词（残基）向量的嵌入维度')
    parse.add_argument('-num-layer', type=int, default=1, help='Transformer的Encoder模块的堆叠层数')
    parse.add_argument('-num-head', type=int, default=4, help='多头注意力机制的头数')
    parse.add_argument('-dim-feedforward', type=int, default=16, help='词（残基）向量的嵌入维度')
    parse.add_argument('-dim-k', type=int, default=8, help='k/q向量的嵌入维度')
    parse.add_argument('-dim-v', type=int, default=8, help='v向量的嵌入维度')

    config = parse.parse_args()
    return config
