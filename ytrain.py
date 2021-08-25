import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from utils.aligning import bp_align
from utils.embedding import OneHotEmbedding
from model import *
from utils.dash2mat import dash2matrix
from Getdataset import IntelDNADataset
from _loss import *



writer = SummaryWriter('./Logs')

# Definition of Embedding Function
Embedder = OneHotEmbedding(4)

# Hyper-Parameters
batch_size = 32
lr=0.001
global_length = 100
n_epoch = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definition of model utils
MODEL = RCNNModel(Block, 8, 2, global_length)
MODEL = MODEL.to(device)
opti = torch.optim.Adam(MODEL.parameters(), lr=lr)
criterion = f1_loss

# Create dataset
trainSet = IntelDNADataset('./dataset_50100', 'train')
testSet = IntelDNADataset('./dataset_50100','test')
train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=True, drop_last=True)

step = 0

for epoch in range(n_epoch):

    name = 'ModelState'

    print("Epoch{}".format(epoch))

    MODEL.train()

    for data in tqdm(train_dataloader):

        # Load data
        bp, label = data

        # Data aligning
        align_bp, align_label = bp_align(bp, label, batch_size)

        # One hot encoding
        encode_bp = [0 for x in range(batch_size)]
        encode_label = torch.zeros([batch_size, global_length, global_length])
        for i in range(batch_size):
            encode_bp[i] = Embedder.encode(align_bp[i])
            encode_label[i] = dash2matrix(align_label[i], global_length)
        encode_bp = torch.tensor(encode_bp)
        encode_bp = encode_bp.permute(0, 2, 1)

        encode_bp = encode_bp.to(device)
        encode_label = encode_label.to(device)

        # Prediction
        out = MODEL(encode_bp)

        opti.zero_grad()

        loss = criterion(out, encode_label)

        loss.backward()

        opti.step()

        step += 1

        writer.add_scalar('Loss', loss, global_step=step)

    MODEL.eval()

    # for data in tqdm(test_dataloader):
    #
    #     bp, label = data
    #
    #     align_bp, align_label = bp_align(bp, label, batch_size)
    #
    #     encode_bp = [0 for x in range(batch_size)]
    #     encode_label = torch.zeros([batch_size, global_length, global_length])
    #     for i in range(batch_size):
    #         encode_bp[i] = Embedder.encode(align_bp[i])
    #         encode_label[i] = dash2matrix(align_label[i], global_length)
    #     encode_bp = torch.tensor(encode_bp)
    #     encode_bp = encode_bp.permute(0, 2, 1)
    #
    #     encode_bp = encode_bp.to(device)
    #     encode_label = encode_label.to(device)
    #
    #     with torch.no_grad():
    #
    #         out = MODEL(encode_bp)
    #
    #         test_loss = criterion(out, encode_label)
    #
    #         step += 1
    #
    #     writer.add_scalar('Test Loss', test_loss, global_step=step)

    name = name + str(n_epoch) + '.pth'
    torch.save(MODEL.state_dict(), name)

writer.close()
