import torch
import torch.nn as nn

from model import bert

'''Classification DNA bert 模型'''
class ClassificationBERT(nn.Module):
    def __init__(self, config):
        super(ClassificationBERT,self).__init__()
        self.config = config

        # 加载预训练模型参数
        self.BERT = bert.BERT(config)

        self.classification = nn.Sequential(
            nn.Linear(768, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 23)
        )

    def forward(self, seqs):
        representation = self.BERT(seqs)["pooler_output"]

        output = self.classification(representation)

        return output, representation