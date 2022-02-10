import torch
import torch.nn as nn

from model import bert, TextCNN
from transformers import BertTokenizer, BertConfig, BertModel

'''bert Fusion 模型'''
class FusionBERT(nn.Module):
    def __init__(self, config):
        super(FusionBERT,self).__init__()
        self.config = config

        self.bertone = bert.BERT(self.config)
        self.berttwo = bert.BERT(self.config)

        self.Ws = torch.nn.Parameter(torch.randn(1, 768).cuda())
        self.Wh = torch.nn.Parameter(torch.randn(1, 768).cuda())


        self.classification = nn.Sequential(
            nn.Linear(768, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 23)
        )

    def forward(self, names, addresses):

        # x = x.cuda()
        # print(seqs)
        representationX = self.bertone(names)["pooler_output"]
        # print(x.shape)
        # representationX = self.TextCNN(x)
        representationY = self.berttwo(addresses)["pooler_output"]

        # print(representationX.shape)
        # print(representationY)

        F = torch.sigmoid(self.Ws * representationX + self.Wh * representationX)

        # print(F)
        representation = F * representationX + (1 - F) * representationY
        # print(representationY.shape)
        # representation = torch.cat((representationX, representationY), dim=1)

        # print(representation.shape)

        output = self.classification(representation)

        return output, representation
