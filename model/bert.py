import sys
# sys.path.append('../')
import os
import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")

'''DNA bert 模型'''
class BERT(nn.Module):
    def __init__(self, config):
        super(BERT,self).__init__()
        self.config = config
        self.config.cuda = True
        # 加载预训练模型参数
        self.pretrainPath = '../pretrain/bert-base-chinese'

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainPath)
        self.bert = BertModel.from_pretrained(self.pretrainPath)

    def forward(self, seqs):
        # print(seqs)
        seqs = list(seqs)
        token_seq = self.tokenizer(seqs, return_tensors='pt', padding=True)
        # print(token_seq)
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        if self.config.cuda:
            representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())
        else:
            representation = self.bert(input_ids, token_type_ids, attention_mask)

        return representation



if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(description='common config')
    config = parse.parse_args()

    model = BERT(config).cuda()
    x = ('莲花池西路北京市海淀区培英小学南侧约150米', '东棉花胡同39号')

    model.train()
    output = model(x)['pooler_output']
    # print(encoded_input)
    # ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    # print(ids)
    # output  = model(**encoded_input)
    print(output)
    print(output.shape)