import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.args = args

        vocab_size = args.vocab_size
        dim_embedding = args.dim_embedding
        num_class = args.num_class
        hidden_size = args.hidden_size
        num_layers = args.num_layers
        bidirectional = args.bidirectional
        num_direction = 2 if bidirectional else 1
        rnn_net = args.rnn_net

        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        if rnn_net == 'RNN':
            self.rnn_net = nn.RNN(input_size=dim_embedding,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  bidirectional=bidirectional)
        elif rnn_net == 'LSTM':
            self.rnn_net = nn.LSTM(input_size=dim_embedding,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional)
        else:
            raise RuntimeError('No such args.rnn_net')

        self.h0 = nn.Parameter(torch.zeros(num_layers * num_direction, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(num_layers * num_direction, 1, hidden_size))

        # self.linear = nn.Linear(hidden_size * num_direction, num_class)
        self.linear = nn.Linear(hidden_size * num_direction * num_layers, num_class)

    def forward(self, x):
        lengths = []
        for seq in x:
            lengths.append(seq.size(0) - np.bincount(seq.cpu())[0])

        lengths = torch.tensor(lengths)
        sort = torch.sort(lengths, descending=True)
        lengths = lengths[sort.indices]
        x = x[sort.indices]
        x = self.embedding(x)
        x = x.transpose(0, 1)

        packed_x = pack_padded_sequence(x, lengths, batch_first=False)
        h0 = self.h0.expand(-1, x.size(1), -1).contiguous()
        c0 = self.c0.expand(-1, x.size(1), -1).contiguous()

        if self.args.rnn_net == 'LSTM':
            pack_out, (hn, cn) = self.rnn_net(packed_x, (h0, c0))
        elif self.args.rnn_net == 'RNN':
            pack_out, hn = self.rnn_net(packed_x, h0)
        else:
            raise RuntimeError('No such args.RNN_net')

        hidden = [hn[i] for i in range(hn.size(0))]
        embedding = torch.cat(hidden, dim=1)
        logits = self.linear(embedding)
        return logits, embedding
