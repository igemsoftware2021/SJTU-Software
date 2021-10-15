import torch
from torch import nn
from model.attention_layer import *
from model.Graph_Layer import *
from utils.GraphTools import *
from utils.outcat import *

class MODELA(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELA, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6 * 2),
            nn.ELU(),
            nn.Conv1d(d_model * 6 * 2 , 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model * 6 * 2, 32, 16, 8, 0.25, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 48, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 2),
            nn.BatchNorm1d(length * 2),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 2)
        return F.softmax(x, dim=2)

class MODELB(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELB, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6 * 2),
            nn.ELU(),
            nn.Conv1d(d_model * 6 * 2 , 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model * 6 * 2, 32, 16, 8, 0.25, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 48, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 3),
            nn.BatchNorm1d(length * 3),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 3)
        return F.softmax(x, dim=2)

class MODELC(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELC, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6 * 2),
            nn.ELU(),
            nn.Conv1d(d_model * 6 * 2 , 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ELU(),
        )
        self.GraphNN = GNNC(d_model * 6 * 2, 128, 64, 32, 16, 0.25, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 48, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 2),
            nn.BatchNorm1d(length * 2),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 2)
        return F.softmax(x, dim=2)

class MODELD(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELD, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6 * 2),
            nn.ELU(),
            nn.Conv1d(d_model * 6 * 2 , 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ELU(),
        )
        self.GraphNN = GNNC(d_model * 6 * 2, 128, 64, 32, 16, 0.25, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 48, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 3),
            nn.BatchNorm1d(length * 3),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 3)
        return F.softmax(x, dim=2)


class MODELE(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELE, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6 * 2),
            nn.ELU(),
            nn.Conv1d(d_model * 6 * 2 , 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model * 6 * 2, 32, 16, 8, 0.25, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 48, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(length * 32, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 32, length * 64),
            nn.BatchNorm1d(length * 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 64, length * 64),
            nn.BatchNorm1d(length * 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 64, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 3),
            nn.BatchNorm1d(length * 3),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 3)
        return F.softmax(x, dim=2)

class MODELF(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELF, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6 * 2),
            nn.ELU(),
            nn.Conv1d(d_model * 6 * 2 , 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model * 6 * 2, 32, 16, 8, 0.25, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 48, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(length * 32, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 32, length * 64),
            nn.BatchNorm1d(length * 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 64, length * 64),
            nn.BatchNorm1d(length * 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 64, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 2),
            nn.BatchNorm1d(length * 2),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 2)
        return F.softmax(x, dim=2)


class SJTURNA(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(SJTURNA, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(2),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model * 6 * 2, 32, 16, 8, 0.25, batch_size)
        self.CovLayer = nn.Sequential(
            nn.Conv2d((d_model * 6 * 2 + 16) * 2, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.MaxPool2d(kernel_size=15, padding=5, stride=5),
            nn.BatchNorm2d(d_model * 6 * 2),
            nn.ELU(),
            nn.Dropout(droprate),
            nn.Conv2d(d_model * 6 * 2, d_model * 2, kernel_size=9, padding=4),
            nn.MaxPool2d(kernel_size=8, padding=3, stride=2),
            nn.BatchNorm2d(d_model * 2),
            nn.ELU(),
        )
        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 64, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 2),
            nn.BatchNorm1d(length * 2),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):
        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        attention_tensor = x
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        # x = x.permute(0, 2, 1)
        # # B, d, L
        # x = self.ResLayer_1d(x)
        # x = self.Cov1d(x)
        # x = x.permute(0, 2, 1)
        # x = F.softmax(x, dim=2)
        # x = outer_cat(x)
        # x = x.permute(0, 3, 1, 2)
        # x = self.CovLayer(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 2)
        return F.softmax(x, dim=2)

class DeepSJTURNA(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(DeepSJTURNA, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6 * 2),
            nn.ELU(),
            nn.Conv1d(d_model * 6 * 2, d_model * 6, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model * 6 * 2, 32, 16, 8, 0.25, batch_size)
        self.CovLayer = nn.Sequential(
            nn.Conv2d((d_model * 6 * 2 + 16) * 2, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.MaxPool2d(kernel_size=15, padding=5, stride=5),
            nn.BatchNorm2d(d_model * 6 * 2),
            nn.ELU(),
            nn.Dropout(droprate),
            nn.Conv2d(d_model * 6 * 2, d_model * 2, kernel_size=9, padding=4),
            nn.MaxPool2d(kernel_size=8, padding=3, stride=2),
            nn.BatchNorm2d(d_model * 2),
            nn.ELU(),
        )
        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 24, length * 48),
            nn.BatchNorm1d(length * 48),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(length * 48, length * 48),
            nn.BatchNorm1d(length * 48),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 48, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 32, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 2),
            nn.BatchNorm1d(length * 2),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):
        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        attention_tensor = x
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # # B, d, L
        x = self.ResLayer_1d(x)
        x = self.Cov1d(x)
        x = x.permute(0, 2, 1)
        # x = F.softmax(x, dim=2)
        # x = outer_cat(x)
        # x = x.permute(0, 3, 1, 2)
        # x = self.CovLayer(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 3)
        return F.softmax(x, dim=2)

class SJTURNA3(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size):
        super(SJTURNA3, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, 3, kernel_size=9, padding=4),
            nn.BatchNorm1d(3),
            nn.ELU(),
        )

        self.GraphNN = GNN(d_model * 6 * 2, 32, 16, 8, 0.25, batch_size)

    def forward(self, x, adj, src_padding_mask):
        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        attention_tensor = x
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.ResLayer_1d(x)
        x = self.Cov1d(x)
        x = x.permute(0, 2, 1)
        x = F.softmax(x, dim=2)
        return x

class MODELH(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELH, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6 * 2),
            nn.ELU(),
            nn.Conv1d(d_model * 6 * 2 , 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model * 6 * 2, 32, 16, 8, 0.25, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 3),
            nn.BatchNorm1d(length * 3),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 3)
        return F.softmax(x, dim=2)


class MODELI(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELI, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=0.2)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model * 6 * 2 + 16, d_model * 6 * 2, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model * 6 * 2),
            nn.ELU(),
            nn.Conv1d(d_model * 6 * 2 , 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model * 6 * 2, 32, 16, 8, 0.25, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 2),
            nn.BatchNorm1d(length * 2),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # L, B, d
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.BiLSTM1(x)
        x = self.BiLSTM2(x)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 2)
        return F.softmax(x, dim=2)

class MODELO(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELO, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=droprate)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model + 16, d_model + 8, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model + 8),
            nn.ELU(),
            nn.Conv1d(d_model + 8, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model, 32, 16, 8, droprate, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 3),
            nn.BatchNorm1d(length * 3),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # B, L, d
        x = x.permute(1, 0, 2)
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 3)
        return F.softmax(x, dim=2)

class MODELO1(nn.Module):
    def __init__(self, d_model, nhead, num_layer, layer_number, batch_size, droprate, length):
        super(MODELO1, self).__init__()
        self.Transformer = Attention_Encoder(d_model, nhead, num_layer)
        self.BiLSTM1 = BiLSTM(d_model, d_model * 2)
        self.BiLSTM2 = BiLSTM(d_model * 4, d_model * 6)
        self.d_model = d_model
        layers_new = []
        for i in range(layer_number):
            al = ResNet1dBlock(in_channel=d_model * 6 * 2 + 16, drop_rate=droprate)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model + 16, d_model + 8, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model + 8),
            nn.ELU(),
            nn.Conv1d(d_model + 8, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
        )
        self.GraphNN = GNN(d_model, 32, 16, 8, droprate, batch_size)

        self.Predit_layer = nn.Sequential(
            nn.Linear(length * 32, length * 64),
            nn.BatchNorm1d(length * 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 64, length * 64),
            nn.BatchNorm1d(length * 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 64, length * 128),
            nn.BatchNorm1d(length * 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 128, length * 64),
            nn.BatchNorm1d(length * 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 64, length * 32),
            nn.BatchNorm1d(length * 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 32, length * 16),
            nn.BatchNorm1d(length * 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 16, length * 8),
            nn.BatchNorm1d(length * 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(length * 8, length * 3),
            nn.BatchNorm1d(length * 3),
            nn.ReLU(),
            nn.Dropout()
        )
    def forward(self, x, adj, src_padding_mask):

        # B, L, d
        x = x.permute(1, 0, 2)
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 0, 2)
        # Output of attention layer
        attention_tensor = x
        # GNN Model Embedding
        GNNEmb = self.GraphNN(attention_tensor, adj)
        # B, L, d
        x = torch.cat((x, GNNEmb), dim=2)
        x = x.permute(0, 2, 1)
        # B, d, L
        x = self.Cov1d(x)
        x = x.reshape(attention_tensor.shape[0], -1)
        x = self.Predit_layer(x)
        x = x.view(attention_tensor.shape[0], -1, 3)
        return F.softmax(x, dim=2)

class ModelP(nn.Module):
    def __init__(self, d_model, attn_layers, attn_nheads, res_layers, drop_rate, global_length):
        super(ModelP, self).__init__()

        self.d_model = d_model
        self.attn_layers = attn_layers
        self.res_layers = res_layers
        self.attn_nheads = attn_nheads
        self.drop_rate = drop_rate
        self.length = global_length

        self.Transformer = Attention_Encoder(self.d_model, self.attn_nheads, self.attn_layers)
        layers_new = []
        for i in range(self.res_layers):
            al = ResNet1dBlock(in_channel=d_model, drop_rate=self.drop_rate)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )
        self.Cov1d = nn.Sequential(
            nn.Conv1d(d_model, 8, kernel_size=9, padding=4),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.Conv1d(8, 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(4),
            nn.ELU(),
        )

        self.Cov2d = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=9, padding=4),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 4, kernel_size=9, padding=4),
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.Conv2d(4, 2, kernel_size=9, padding=4),
            nn.ELU(),
            nn.Conv2d(2, 1, kernel_size=9, padding=4),
        )

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

    def forward(self, x, src_padding_mask):
        x = self.Transformer(x, src_padding_mask)
        x = x.permute(1, 2, 0)
        x = self.ResLayer_1d(x)
        x = self.Cov1d(x)
        x = self.matrix_rep(x)
        x = self.Cov2d(x)
        x = x.view(-1, self.length, self.length)
        x = F.softmax(x, dim=2)
        return x





