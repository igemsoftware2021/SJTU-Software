import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.outcat import *
from Encoder_Layers import MYTransformerEncoderLayer


class Block(nn.Module):
    def __init__(self, in_channel, drop_rate):
        super(Block, self).__init__()
        self.Cov1 = nn.Conv2d(in_channel, in_channel*2, 3, stride=1, padding=1)
        self.Cov2 = nn.Conv2d(in_channel*2, in_channel*2, 5, stride=1, padding=2)
        self.Cov3 = nn.Conv2d(in_channel, in_channel*2, 1, stride=1)
        self.Block = nn.Sequential(
            self.Cov1,
            nn.BatchNorm2d(in_channel*2),
            nn.ELU(),
            nn.Dropout(drop_rate),
            self.Cov2,
            nn.BatchNorm2d(in_channel*2),
            nn.ELU(),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        y = self.Block(x)
        x = self.Cov3(x)
        return F.relu(x + y)


class RCNNModel(nn.Module):
    def __init__(self, Block, in_channel, layer_number, max_len, drop_rate=0.25):
        super(RNAModel, self).__init__()
        self.prenet = nn.Conv2d(in_channel, 16, 3, stride=1, padding=1)
        # self.cov1 = nn.Conv1d(4, 4, 3, stride=1, padding=1)
        # self.cov2 = nn.Conv1d(4, 4, 5, stride=1, padding=2)
        # self.block = nn.Sequential(
        #     self.cov1,
        #     nn.BatchNorm1d(4),
        #     nn.ELU(),
        #     nn.Dropout(drop_rate),
        #     self.cov2,
        #     nn.BatchNorm1d(4),
        #     nn.ELU(),
        #     nn.Dropout(drop_rate)
        # )
        # layers_new = []
        # for i in range(layer_number):
        #     cl = self.block
        #     setattr(self, "cl%i" % i, cl)
        #     layers_new.append(cl)
        # self.ResLayer_1d = nn.Sequential(
        #     *layers_new
        # )

        self.Encoder = nn.TransformerEncoderLayer(4, 1)
        self.Encoder_Layer = nn.TransformerEncoder(self.Encoder, num_layers=1)
        self.max_len = max_len
        self.layer_number = layer_number
        layers = []
        for i in range(self.layer_number):
            bl = Block(8 * 2**(i), drop_rate)
            setattr(self, "bl%i" % i, bl)
            layers.append(bl)
        self.ResLayer = nn.Sequential(
            *layers
        )
        self.AvergerNet = nn.Sequential(
            nn.AdaptiveMaxPool2d((20,20)),
            nn.Flatten()
        )
        self.preditlayer = nn.Sequential(
            nn.Linear(8 * 2 ** self.layer_number*400,
                      self.max_len * self.max_len),
            nn.Sigmoid()
        )

    def getlen(self, x):
        self.max_len = x

    def matrix_rep(self, x):
        '''
        for each position i,j of the matrix, we concatenate the embedding of i and j
        '''
        x = x.permute(0, 2, 1)  # L*d
        L = x.shape[1]
        x2 = x
        x = x.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        x = x.repeat(1, L, 1, 1)
        x2 = x2.repeat(1, 1, L, 1)
        mat = torch.cat([x, x2], -1)  # L*L*2d

        # make it symmetric
        # mat_tril = torch.cat(
        #     [torch.tril(mat[:,:, i]) for i in range(mat.shape[-1])], -1)
        mat_tril = torch.tril(mat.permute(0, -1, 1, 2))  # 2d*L*L
        mat_diag = mat_tril - torch.tril(mat.permute(0, -1, 1, 2), diagonal=-1)
        mat = mat_tril + torch.transpose(mat_tril, -2, -1) - mat_diag

        return mat

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # x = self.ResLayer_1d(x)
        # x = x.permute(0, 2, 1)
        x = self.Encoder_Layer(x)
        x = self.matrix_rep(x)
        x = x.permute(0, 3, 1, 2)
        #x = self.prenet(x)
        x = self.ResLayer(x)
        x = self.AvergerNet(x)
        x = self.preditlayer(x)
        return x.view(-1, self.max_len, self.max_len)


class RNAModel(nn.Module):
    def __init__(self, Block, in_channel, layer_number, max_len, drop_rate=0.25):
        super(RNAModel, self).__init__()
        self.prenet = nn.Conv2d(in_channel, 16, 3, stride=1, padding=1)
        self.Encoder = nn.TransformerEncoderLayer(4, 1)
        self.Encoder_Layer = nn.TransformerEncoder(self.Encoder, num_layers=1)
        self.max_len = max_len
        self.layer_number = layer_number
        layers = []
        for i in range(self.layer_number):
            bl = Block(16 * 2**(i), drop_rate)
            setattr(self, "bl%i" % i, bl)
            layers.append(bl)
        self.ResLayer = nn.Sequential(
            *layers
        )
        self.AvergerNet = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten()
        )
        self.preditlayer = nn.Sequential(
            nn.Linear(16 * 2 ** self.layer_number, self.max_len * self.max_len),
            nn.Sigmoid()
        )

    def getlen(self, x):
        self.max_len = x


    def forward(self, x):
        x = self.Encoder_Layer(x)
        x = outer_cat(x)
        x = x.permute(0, 3, 1, 2)
        #x = x.to('cuda')
        x = self.prenet(x)
        x = self.ResLayer(x)
        x = self.AvergerNet(x)
        x = self.preditlayer(x)
        return x.view(-1, self.max_len, self.max_len)

class TagModel150(nn.Module):

    def __init__(self, dim, resnet_layer, layer_number, drop_rate=0.25):
        super(TagModel150, self).__init__()
        self.dim = dim
        self.layer_number = layer_number
        self.encode_layer = nn.TransformerEncoderLayer(self.dim, nhead=1)
        self.attention_layer = nn.TransformerEncoder(self.encode_layer, num_layers=1)
        self.channel_attention = nn.TransformerEncoderLayer(d_model=16 * 2 ** self.layer_number, nhead=1)
        self.channel_attention_layer = nn.TransformerEncoder(self.channel_attention, num_layers=1)
        self.prenet = nn.Conv2d(dim*2, 16, 3, stride=1, padding=1)
        self.resnet_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((25, 25))
        )
        self.seq_cov = nn.Sequential(
            nn.Conv1d(16 * 2 ** self.layer_number, 8 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(8 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(8 * 2 ** self.layer_number, 4 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(4 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(4 * 2 ** self.layer_number, 2 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=4),
            nn.Conv1d(2 * 2 ** self.layer_number, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(kernel_size=6, stride=2, padding=1, dilation=3)
        )

        layers = []
        for i in range(self.layer_number):
            bl = resnet_layer(16 * 2 ** (i), drop_rate)
            setattr(self, "bl%i" % i, bl)
            layers.append(bl)
        self.ResLayer = nn.Sequential(
            *layers
        )

    def forward(self, x):
        x = self.attention_layer(x)
        x = outer_cat(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.prenet(x)
        x = self.ResLayer(x)
        x = self.resnet_pool(x)
        x = x.view(x.shape[0], -1, x.shape[2]**2)
        x = x.permute(0, 2, 1).contiguous()
        x = self.channel_attention_layer(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.seq_cov(x)
        return x.permute(0, 2, 1).contiguous()


class TagModel100(nn.Module):

    def __init__(self, dim, resnet_layer, layer_number, drop_rate=0.25):
        super(TagModel100, self).__init__()
        self.dim = dim
        self.layer_number = layer_number
        self.encode_layer = nn.TransformerEncoderLayer(self.dim, nhead=1)
        self.attention_layer = nn.TransformerEncoder(self.encode_layer, num_layers=1)
        self.channel_attention = nn.TransformerEncoderLayer(d_model=16 * 2 ** self.layer_number, nhead=1)
        self.channel_attention_layer = nn.TransformerEncoder(self.channel_attention, num_layers=1)
        self.prenet = nn.Conv2d(dim*2, 16, 3, stride=1, padding=1)
        self.resnet_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((25, 25))
        )
        self.seq_cov = nn.Sequential(
            nn.Conv1d(16 * 2 ** self.layer_number, 8 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(8 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(8 * 2 ** self.layer_number, 4 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(4 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(4 * 2 ** self.layer_number, 2 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=4),
            nn.Conv1d(2 * 2 ** self.layer_number, 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(3),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(kernel_size=5, stride=3, padding=1, dilation=4)
        )

        layers = []
        for i in range(self.layer_number):
            bl = resnet_layer(16 * 2 ** (i), drop_rate)
            setattr(self, "bl%i" % i, bl)
            layers.append(bl)
        self.ResLayer = nn.Sequential(
            *layers
        )

    def forward(self, x):
        x = self.attention_layer(x)
        x = outer_cat(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.prenet(x)
        x = self.ResLayer(x)
        x = self.resnet_pool(x)
        x = x.view(x.shape[0], -1, x.shape[2]**2)
        x = x.permute(0, 2, 1).contiguous()
        x = self.channel_attention_layer(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.seq_cov(x)
        return x.permute(0, 2, 1).contiguous()


class Binary_TagModel_150(nn.Module):

    def __init__(self, dim, resnet_layer, layer_number, drop_rate=0.25):
        super(Binary_TagModel_150, self).__init__()
        self.dim = dim
        self.layer_number = layer_number
        self.encode_layer = nn.TransformerEncoderLayer(self.dim, nhead=1)
        self.attention_layer = nn.TransformerEncoder(self.encode_layer, num_layers=1)
        self.channel_attention = nn.TransformerEncoderLayer(d_model=16 * 2 ** self.layer_number, nhead=1)
        self.channel_attention_layer = nn.TransformerEncoder(self.channel_attention, num_layers=1)
        self.prenet = nn.Conv2d(dim*2, 16, 3, stride=1, padding=1)
        self.resnet_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((25, 25))
        )
        self.seq_cov = nn.Sequential(
            nn.Conv1d(16 * 2 ** self.layer_number, 8 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(8 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(8 * 2 ** self.layer_number, 4 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(4 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(4 * 2 ** self.layer_number, 2 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=4),
            nn.Conv1d(2 * 2 ** self.layer_number, 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(2),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(kernel_size=6, stride=2, padding=1, dilation=3)
        )

        layers = []
        for i in range(self.layer_number):
            bl = resnet_layer(16 * 2 ** (i), drop_rate)
            setattr(self, "bl%i" % i, bl)
            layers.append(bl)
        self.ResLayer = nn.Sequential(
            *layers
        )

    def forward(self, x):
        x = self.attention_layer(x)
        x = outer_cat(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.prenet(x)
        x = self.ResLayer(x)
        x = self.resnet_pool(x)
        x = x.view(x.shape[0], -1, x.shape[2]**2)
        x = x.permute(0, 2, 1).contiguous()
        x = self.channel_attention_layer(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.seq_cov(x)
        return x.permute(0, 2, 1).contiguous()


class Binary_TagModel_100(nn.Module):

    def __init__(self, dim, resnet_layer, layer_number, drop_rate=0.25):
        super(Binary_TagModel_100, self).__init__()
        self.dim = dim
        self.layer_number = layer_number
        self.encode_layer = nn.TransformerEncoderLayer(self.dim, nhead=4)
        self.attention_layer = nn.TransformerEncoder(self.encode_layer, num_layers=2)
        self.channel_attention = nn.TransformerEncoderLayer(d_model=16 * 2 ** self.layer_number, nhead=16)
        self.channel_attention_layer = nn.TransformerEncoder(self.channel_attention, num_layers=2)
        self.prenet = nn.Conv2d(dim*2, 16, 3, stride=1, padding=1)
        self.resnet_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((25, 25))
        )
        self.seq_cov = nn.Sequential(
            nn.Conv1d(16 * 2 ** self.layer_number, 8 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(8 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(8 * 2 ** self.layer_number, 4 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(4 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(4 * 2 ** self.layer_number, 2 * 2 ** self.layer_number, kernel_size=3, padding=1),
            nn.BatchNorm1d(2 * 2 ** self.layer_number),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(kernel_size=10, stride=2, padding=4),
            nn.Conv1d(2 * 2 ** self.layer_number, 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(2),
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.MaxPool1d(kernel_size=5, stride=3, padding=1, dilation=4)
        )

        layers = []
        for i in range(self.layer_number):
            bl = resnet_layer(16 * 2 ** (i), drop_rate)
            setattr(self, "bl%i" % i, bl)
            layers.append(bl)
        self.ResLayer = nn.Sequential(
            *layers
        )

    def forward(self, x):
        x = self.attention_layer(x)
        x = outer_cat(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.prenet(x)
        x = self.ResLayer(x)
        x = self.resnet_pool(x)
        x = x.view(x.shape[0], -1, x.shape[2]**2)
        x = x.permute(0, 2, 1).contiguous()
        x = self.channel_attention_layer(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.seq_cov(x)
        return x.permute(0, 2, 1).contiguous()
