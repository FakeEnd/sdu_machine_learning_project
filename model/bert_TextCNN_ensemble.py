import torch
import torch.nn as nn
from model import TextCNN, TextRNN, RNN_Attention, bert


class Ensemble(nn.Module):
    def __init__(self, config):
        super(Ensemble, self).__init__()
        self.config = config

        self.TextCNN = TextCNN.TextCNN(config)
        self.bert = bert.BERT(config)

        num_class = 23
        # dim_embedding = args.CNN_dim_embedding * 8 + args.RNN_dim_embedding * 2 * 2 + args.d_model
        dim_embedding = 128 + 768

        # print('dim_embedding', dim_embedding)
        self.linear1 = nn.Linear(dim_embedding, dim_embedding // 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.linear2 = nn.Linear(dim_embedding // 2, num_class)

    def forward(self, data):
        _, TextCNN_embedding = self.TextCNN(data)
        bert_embedding = self.bert(data)["pooler_output"]


        # embeddings = [embedding1, embedding2, embedding3, embedding4]
        embeddings = [TextCNN_embedding, bert_embedding]

        # print('input_ids', input_ids.size())
        # print('embedding1', embedding1.size())
        # print('embedding2', embedding2.size())

        embeddings = torch.cat(embeddings, dim=-1)
        # print('embeddings', embeddings.size())

        out = self.linear1(embeddings)
        out = self.leakyrelu(out)
        logits = self.linear2(out)
        return logits, bert_embedding
