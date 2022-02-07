# ---encoding:utf-8---
# @Time : 2020.11.09
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : BERT_relu.py


import torch
import torch.nn as nn
import numpy as np
from configur import config
import pickle

'''
模型构建
'''


# 目的是构造出一个注意力判断矩阵，一个[batch_size, seq_len, seq_len]的张量
# 其中参与注意力计算的位置被标记为TRUE，将token为[PAD]的位置掩模标记为FALSE
def get_attn_pad_mask(seq):
    # 在BERT中由于是self_attention，seq_q和seq_k内容相同
    # print('-' * 50, '掩模', '-' * 50)

    batch_size, seq_len = seq.size()
    # print('batch_size', batch_size)
    # print('seq_len', seq_len)

    # print('-' * 10, 'test', '-' * 10)
    # print(seq_q.data.shape)
    # print(seq_q.data.eq(0).shape)
    # print(seq_q.data.eq(0).unsqueeze(1).shape)

    # seq_q.data取出张量seq_q的数据
    # seq_q.data.eq(0)是一个和seq_q.data相同shape的张量，seq_q.data对应位置为0时，结果的对应位置为TRUE，否则为FALSE
    # eq(zero) is PAD token 如果等于0，证明那个位置是[PAD]，因此要掩模，计算自注意力时不需要考虑该位置
    # unsqueeze(1)是在维度1处插入一个维度，维度1及其之后的维度往后移，从原来的[batch_size, seq_len]变成[batch_size, 1, seq_len]
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]

    # expand是将某一个shape为1的维度复制为自己想要的数量，这里是从[batch_size, 1, seq_len]将1维度复制seq_len份
    # 结果是变成[batch_size, seq_len, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

    return pad_attn_mask_expand


def repeat_by_dim(tensor, num=6, dim=1):
    tensor = tensor.unsqueeze(dim)
    tensor = [tensor] * num
    tensor = torch.cat(tensor, dim=dim)
    return tensor


# 嵌入层
class Embedding(nn.Module):
    def __init__(self, user_embedding, road_map_embedding, transport_map_embedding):
        super(Embedding, self).__init__()
        self.user_embed = nn.Embedding(user_vocab_size, d_model)  # user embedding
        self.pos_embed = nn.Embedding(pos_vocab_size, d_model)  # position embedding
        self.time_embed = nn.Embedding(time_vocab_size, d_model)  # time embedding

        user_embedding = np.array(user_embedding)
        user_embedding = torch.from_numpy(user_embedding)
        road_map_embedding = torch.from_numpy(road_map_embedding)
        transport_map_embedding = torch.from_numpy(transport_map_embedding)

        # 补4个tensor
        pad_tensor_1 = torch.zeros([4, 32])
        pad_tensor_2 = torch.zeros([4, 32])
        road_map_embedding = torch.cat([pad_tensor_1, road_map_embedding], dim=0)
        transport_map_embedding = torch.cat([pad_tensor_2, transport_map_embedding], dim=0)

        print('road_map_embedding.Embedding.size()', [pos_vocab_size, d_model])
        print('road_map_embedding.size()', road_map_embedding.size())
        print('transport_map_embedding.size()', transport_map_embedding.size())
        print('user_embedding.size()', user_embedding.size())

        print('road_map_embedding', road_map_embedding[:5])
        print('transport_map_embedding', transport_map_embedding[:5])

        self.doc2vec_user_embedding = nn.Embedding(user_vocab_size, d_model).from_pretrained(
            user_embedding)
        self.road_map_embedding = nn.Embedding(pos_vocab_size, d_model).from_pretrained(
            road_map_embedding)
        self.transport_map_embedding = nn.Embedding(pos_vocab_size, d_model).from_pretrained(
            transport_map_embedding)

        self.doc2vec_user_embedding.requires_grad_(True)
        self.road_map_embedding.requires_grad_(True)
        self.transport_map_embedding.requires_grad_(True)

        self.norm = nn.LayerNorm(d_model * 3 + 64)

    def forward(self, user, pos, time):
        user_embed = self.user_embed(user)
        pos_embed = self.pos_embed(pos)
        time_embed = self.time_embed(time)
        user_embed = repeat_by_dim(user_embed, pos_embed.size(1), dim=1)

        doc2vec_user_embedding = self.doc2vec_user_embedding(user)
        doc2vec_user_embedding = repeat_by_dim(doc2vec_user_embedding, pos_embed.size(1), dim=1)

        road_map_embed = self.road_map_embedding(pos)
        transport_map_embed = self.transport_map_embedding(pos)

        # embedding = torch.cat([time_embed, pos_embed, doc2vec_user_embedding,
        #                        road_map_embed, transport_map_embed], dim=2)

        # embedding = torch.cat([time_embed, user_embed, doc2vec_user_embedding, pos_embed,
        #                        road_map_embed, transport_map_embed], dim=2)
        #
        embedding = torch.cat(
            [user_embed, time_embed, pos_embed, doc2vec_user_embedding, road_map_embed], dim=2)
        # embedding = torch.cat([user_embed, time_embed, pos_embed], dim=2)
        # print('embedding', embedding.size())

        # layerNorm
        embedding = self.norm(embedding)
        return embedding


# 计算Self-Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        #  这里接收的Q, K, V是q_s, q_k, q_v，也就是真正的Q, K, V是向量，shape:[bach_size, seq_len, d_model]
        # Q: [batch_size, n_head, seq_len, d_k]
        # K: [batch_size, n_head, seq_len, d_k]
        # V: [batch_size, n_head, seq_len, d_v]

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]

        # mask_filled:是将mask中为1/TRUE的元素所在的索引，在原tensor中相同的的索引处替换为指定的value
        # remark: mask必须是一个 ByteTensor而且shape必须和a一样，mask value必须同为tensor
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.

        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model * 3 + 64, d_k * n_head)
        self.W_K = nn.Linear(d_model * 3 + 64, d_k * n_head)
        self.W_V = nn.Linear(d_model * 3 + 64, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model * 3 + 64)
        self.norm = nn.LayerNorm(d_model * 3 + 64)

    def forward(self, Q, K, V, attn_mask):
        # print('Q',Q.size())
        # print('K', Q.size())
        # print('V', Q.size())
        #  这里接收的Q, K, V都是enc_inputs，也就是embedding后的输入，shape:[bach_size, seq_len, d_model]
        # Q: [batch_size, seq_len, d_model]
        # K: [batch_size, seq_len, d_model]
        # V: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # 多头注意力是同时计算的，一次tensor乘法即可，这里是将多头注意力进行切分
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]

        # 处理前attn_mask: [batch_size, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        # 处理后attn_mask: [batch_size, n_head, seq_len, seq_len]

        # context: [batch_size, n_head, seq_len, d_v], attn: [batch_size, n_head, seq_len, seq_len]
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        # print('context.device', context.device)

        # output = nn.Linear(n_head * d_v, d_model)(context)  # context: [batch_size, seq_len, d_model]
        output = self.linear(context)

        # layerNorm
        # output = nn.LayerNorm(d_model)(output + residual)  # context: [batch_size, seq_len, d_model]

        # print('output', output.shape)
        # print('residual', residual.shape)
        output = self.norm(output + residual)
        return output, attention_map


# 基于位置的全连接层
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model * 3 + 64, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model * 3 + 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 多头注意力模块
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        # 全连接模块
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


# 完整的模型
class BERT(nn.Module):
    def __init__(self, config_train):
        super(BERT, self).__init__()

        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, \
            user_vocab_size, pos_vocab_size, time_vocab_size, device
        max_len = config_train.max_len
        n_layers = config_train.num_layer
        n_head = config_train.num_head
        d_model = config_train.dim_embedding
        d_ff = config_train.dim_feedforward
        d_k = config_train.dim_k
        d_v = config_train.dim_v
        user_vocab_size = config_train.user_vocab_size
        pos_vocab_size = config_train.pos_vocab_size
        time_vocab_size = config_train.time_vocab_size
        num_class = config_train.num_class
        device = torch.device("cuda" if config_train.cuda else "cpu")
        print('BERT definition: max_len', max_len)

        # Embedding层
        self.embedding = Embedding(config_train.user_embedding, config_train.road_map_embedding,
                                   config_train.transport_map_embedding)

        # Encoder Block
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 定义重复的模块

        # self.fc_task = nn.Sequential(
        #     nn.Linear(d_model * 3 + 64, d_model),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(d_model, num_class),
        # )
        self.fc_task = nn.Sequential(
            nn.Linear(d_model * 3 + 64, num_class),
        )

    def forward(self, user, pos, time):
        # embedding层
        output = self.embedding(user, pos, time)  # [bach_size, seq_len, d_model*2]
        # print('output', output.size())
        # print('output', output)
        # print('user', user)
        # print('pos', pos)
        # print('time', time)

        # 获取掩模判断矩阵
        enc_self_attn_mask = get_attn_pad_mask(pos)  # [batch_size, max_len, max_len]

        # 逐层通过各个Encoder模块
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
            # output: [batch_size, max_len, d_model]

        # 只取[CLS]
        representation = output[:, 0, :]
        logits_clsf = self.fc_task(representation)
        logits_clsf = logits_clsf.view(logits_clsf.size(0), -1)

        return logits_clsf, representation


def check_model():
    config_train = config.get_train_config()
    torch.cuda.set_device(config_train.device)  # 选择要使用的GPU

    config_train = config.get_train_config()
    config_train.user_vocab_size = 37426 + 4
    config_train.time_vocab_size = 1440 + 4
    config_train.pos_vocab_size = 1881 + 4
    config_train.max_len = 41

    model = BERT(config_train)

    print('-' * 50, 'Model', '-' * 50)
    print(model)

    print('-' * 50, 'Model.named_parameters', '-' * 50)
    for name, value in model.named_parameters():
        print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))

    print('-' * 50, 'Model.named_children', '-' * 50)
    for name, child in model.named_children():
        print('\\' * 40, '[name:{}]'.format(name), '\\' * 40)
        print('child:\n{}'.format(child))

        if name == 'soft_attention':
            print('soft_attention')
            for param in child.parameters():
                print('param.shape', param.shape)
                print('param.requires_grad', param.requires_grad)

        for sub_name, sub_child in child.named_children():
            print('*' * 20, '[sub_name:{}]'.format(sub_name), '*' * 20)
            print('sub_child:\n{}'.format(sub_child))

            # if name == 'layers' and (sub_name == '5' or sub_name == '4'):
            if name == 'layers' and (sub_name == '5'):
                print('Ecoder 5 is unfrozen')
                for param in sub_child.parameters():
                    param.requires_grad = True

        # for param in child.parameters():
        #     print('param.requires_grad', param.requires_grad)

    print('-' * 50, 'Model.named_parameters', '-' * 50)
    for name, value in model.named_parameters():
        print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))


if __name__ == '__main__':
    # check model
    check_model()
