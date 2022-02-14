import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config

        self.filter_sizes = [1, 2, 3, 4, 5, 8]
        self.embedding_dim = 128
        dim_cnn_out = 23
        filter_num = 128

        # 借用一下token编码
        self.pretrainPath = '../pretrain/bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainPath)

        # self.filter_sizes = [int(fsz) for fsz in self.filter_sizes.split(',')]
        self.embedding = nn.Embedding(20000, self.embedding_dim, padding_idx=0)
        # self.embedding = nn.Embedding(69, self.embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, self.embedding_dim)) for fsz in self.filter_sizes])

        self.classification =nn.Linear(128, 23)

        self.linear = nn.Sequential(
            nn.Linear(len(self.filter_sizes) * filter_num, 128),
            nn.ReLU(),
        )

    def forward(self, seqs):

        seqs = list(seqs)
        seqs_ids = self.tokenizer(seqs, return_tensors='pt', padding=True)["input_ids"].cuda()

        x = self.embedding(seqs_ids)  # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)
        # print('embedding x', x.size())

        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.embedding_dim)
        # print('view x', x.size())

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]
        # print(x)
        # print('conv x', len(x), [x_item.size() for x_item in x])

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # print('max_pool2d x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # print('flatten x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        representation_origin = torch.cat(x, 1)
        # print('concat x', x.size()) torch.Size([320, 1024])

        representation = self.linear(representation_origin)

        output = self.classification(representation)

        return output, representation
