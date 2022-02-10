import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN_Attention(nn.Module):
    def __init__(self, args):
        super(RNN_Attention, self).__init__()
        self.args = args

        self.vocab_size = args.vocab_size
        self.dim_embedding = args.dim_embedding
        self.num_class = args.num_class
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.bidirectional = args.bidirectional
        self.num_direction = 2 if self.bidirectional else 1
        self.attention_size = args.attention_size
        self.rnn_net = args.rnn_net

        self.embedding = nn.Embedding(self.vocab_size, self.dim_embedding)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        if self.rnn_net == 'RNN':
            self.rnn = nn.RNN(input_size=self.dim_embedding,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional)
        if self.rnn_net == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.dim_embedding,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional)
        elif self.rnn_net == 'GRU':
            self.rnn = nn.GRU(input_size=self.dim_embedding,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=self.bidirectional)
        else:
            raise RuntimeError('No such args.rnn_net')

        # initial h and c
        self.h0 = nn.Parameter(torch.zeros(self.num_layers * self.num_direction, 1, self.hidden_size))
        self.c0 = nn.Parameter(torch.zeros(self.num_layers * self.num_direction, 1, self.hidden_size))

        # classification
        self.linear = nn.Linear(self.hidden_size * self.num_direction, self.num_class)

        # soft attention
        self.atten = nn.Linear(self.hidden_size * self.num_direction, 1)
        # self.w_omega = nn.Linear(self.hidden_size * self.num_direction, self.attention_size)
        # self.u_omega = nn.Linear(self.attention_size, 1)

    def soft_attention(self, output):
        # lstm_output: (seq_len, batch_size, hidden_size*num_direction)
        seq_len = output.size(0)
        output_reshape = output.reshape([-1, self.hidden_size * self.num_direction])
        # (seq_len * batch_size, hidden_size*layer_size)
        scores = self.atten(output_reshape)  # (seq_len * batch_size, 1)
        scores = scores.reshape([-1, seq_len])
        alphas = scores.softmax(dim=1)  # (batch_size, seq_len)
        alphas_reshape = alphas.unsqueeze(2)  # (batch_size, seq_len, 1)
        state = output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size*layer_size)
        attn_output = torch.sum(state * alphas_reshape, 1)  # (batch_size, hidden_size*layer_size)
        return attn_output

    def attention(self, rnn_out, state):
        # rnn_out: (seq_len, batch, hidden_size*num_direction)
        # state: (num_layers*num_direction, batch, hidden_size*num_direction)
        rnn_out = rnn_out.transpose(0, 1)  # (batch, seq_len, hidden_size*num_direction)
        if self.bidirectional:
            merged_state = torch.cat([state[-2], state[-1]], 1)
        else:
            merged_state = state[-1]
        merged_state = merged_state.unsqueeze(2)  # (batch, hidden_size*num_direction, 1)
        # (batch, seq_len, hidden_size*num_direction) * (batch, hidden_size*num_direction, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, hidden_size*num_direction, seq_len) * (batch, seq_len, 1) = (batch, hidden_size*num_direction, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

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

        if self.rnn_net == 'RNN':
            pack_out, hn = self.rnn(packed_x, h0)
        elif self.rnn_net == 'LSTM':
            pack_out, (hn, cn) = self.rnn(packed_x, (h0, c0))
        elif self.rnn_net == 'GRU':
            pack_out, hn = self.rnn(packed_x, h0)
        else:
            raise RuntimeError('No such args.rnn_net')

        output, others = pad_packed_sequence(pack_out, batch_first=False)
        embedding = self.soft_attention(output)
        # attn_output = self.attention(output, hn)
        logits = self.linear(embedding)
        return logits, embedding
