from model.attention_layer import *
from model.Graph_Layer import *
from utils.outcat import *

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






