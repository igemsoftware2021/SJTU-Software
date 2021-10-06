import torch.nn as nn
from utils.outcat import *

class BiLSTM(nn.Module):
    def __init__(self, input_dim, n_hidden):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, n_hidden, bidirectional=True)

    def forward(self, X):
        # X : [batch_size, seq_len, input_dim]
        input = X.transpose(0, 1)  # input : [seq_len, batch_size, embedding_dim]
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.transpose(0, 1)  # output : [batch_size, seq_len, n_hidden]
        return output

class Block2(nn.Module):
    def __init__(self, in_channel, drop_rate):
        super(Block2, self).__init__()
        self.cov1 = nn.Conv1d(in_channel, in_channel, 3, padding=1)
        self.cov2 = nn.Conv1d(in_channel, in_channel, 5, padding=2)
        self.Block2 = nn.Sequential(
            self.cov1,
            nn.BatchNorm1d(in_channel),
            nn.ELU(),
            nn.Dropout(drop_rate),
            self.cov2,
            nn.BatchNorm1d(in_channel),
            nn.ELU(),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        y = self.Block2(x)
        return x + y


class RNAModel_binary(nn.Module):
    def __init__(self, in_channel, layer_number,max_len,drop_rate=0.5):
        super(RNAModel_binary, self).__init__()
        self.max_len = max_len
        self.layer_number = layer_number

        layers_new = []
        for i in range(layer_number):
            al = Block2(in_channel=in_channel,drop_rate = drop_rate)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )

        self.Encoder = nn.TransformerEncoderLayer(in_channel, 2)
        self.Encoder_Layer = nn.TransformerEncoder(self.Encoder, num_layers=1)

        self.Bi_layer1 = BiLSTM(input_dim=in_channel,n_hidden=in_channel)
        self.Bi_layer2 = BiLSTM(input_dim=in_channel*2,n_hidden=in_channel*2)

        self.preditlayer1 = nn.Sequential(
            nn.Linear(16*self.max_len,
                      512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(drop_rate),
        )

        self.preditlayer2 = nn.Sequential(
            nn.Linear(512,
                      self.max_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.ResLayer_1d(x)
        x = x.permute(0, 2, 1)
        x = self.Encoder_Layer(x)
        x = self.Bi_layer1(x)
        x = self.Bi_layer2(x)
        x = x.reshape(x.shape[0],-1)
        x = self.preditlayer1(x)
        x = self.preditlayer2(x)
        return x.view(-1, self.max_len)
