import torch
import torch.nn as nn
from torch import Tensor
import math
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        pe = pe.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)].to(device)
        return self.dropout(x)


class Attention_Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(Attention_Encoder, self).__init__()
        self.encode_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(self.encode_layer, num_layers)

    def forward(self, x, src_mask):
        return self.encoder(x, mask=None, src_key_padding_mask=src_mask)


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

class ResNet1dBlock(nn.Module):
    def __init__(self, in_channel, drop_rate=0.2):
        super(ResNet1dBlock, self).__init__()
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

class ResNet2dBlock(nn.Module):
    def __init__(self, in_channel, drop_rate=0.2):
        super(ResNet2dBlock, self).__init__()
        self.cov1 = nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.cov2 = nn.Conv2d(in_channel, in_channel, 5, padding=2)
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

