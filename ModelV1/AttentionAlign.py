
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch.nn.functional as F
from utils.embedding import OneHotEmbedding

from Getdataset import IntelDNADataset
from _loss import *

class AttentionAlign(nn.Module):
    def __init__(self, dim, global_length=100):
        super(AttentionAlign, self).__init__()
        self.global_length=global_length
        self.AttentionEncodeLayer = nn.TransformerEncoderLayer(d_model=dim, nhead=1)
        self.AttentionEncoder = nn.TransformerEncoder(self.AttentionEncodeLayer, num_layers=1)
        self.AttentionAlignDecodeLayer = nn.TransformerDecoderLayer(4, nhead=1)
        self.AttentionAlignDecoder = nn.TransformerDecoder(self.AttentionAlignDecodeLayer, num_layers=1)

    def forward(self, x):
        self.origin_length = x.shape[1]
        self.pred = nn.Sequential(
            nn.Linear(self.origin_length * 4, self.global_length * 4),
            nn.ReLU(),
            nn.Linear(self.global_length * 4, self.global_length * 8),
            nn.ReLU(),
            nn.Linear(self.global_length * 8, self.global_length * 4),
        )
        self.decode = nn.Sequential(
            nn.Linear(self.global_length * 4, self.global_length * 8),
            nn.ReLU(),
            nn.Linear(self.global_length * 8, self.global_length * 4),
            nn.ReLU(),
            nn.Linear(self.global_length * 4, self.origin_length * 4)
        )
        x = self.AttentionEncoder(x)
        x = x.flatten()
        x = self.pred(x)
        x = x.view(1, self.global_length, 4)
        y = x
        x = self.AttentionAlignDecoder(x)
        x.flatten()
        x = self.decode(x)
        x = x.view(1, self.origin_length, 4)
        return y, x



writer = SummaryWriter('./Pretrainloss')

# Definition of Embedding Function
Embedder = OneHotEmbedding(4)

# Hyper-Parameters
batch_size = 1
lr=0.002
global_length = 100
n_epoch = 2
device = torch.device('cpu')

# Definition of model utils
MODEL = AttentionAlign(4, global_length)

MODEL = MODEL.to(device)
opti = torch.optim.Adam(MODEL.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

# Create dataset
trainSet = IntelDNADataset('./dataset_full', 'train')
testSet = IntelDNADataset('./dataset_full', 'test')
train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=False, drop_last=True)

train_step = 0

for epoch in range(n_epoch):

    name = 'PretrainModel'

    print("Epoch{}".format(epoch))

    MODEL.train()

    for data in tqdm(train_dataloader):

        # Load data
        bp, _ = data

        # One hot encoding
        encode_bp = [0 for x in range(batch_size)]

        for i in range(batch_size):
            encode_bp[i] = Embedder.encode(bp[i])

        encode_bp = torch.tensor(encode_bp)
        encode_bp = encode_bp.permute(0, 2, 1)
        encode_bp = encode_bp.to(device)

        opti.zero_grad()

        _, out = MODEL(encode_bp)

        loss = criterion(out, encode_bp)

        loss.backward()

        opti.step()

        train_step += 1

        writer.add_scalar('Loss', loss, global_step=train_step)

    name = name + str(epoch) + '.pth'
    torch.save(MODEL.state_dict(), name)

writer.close()

