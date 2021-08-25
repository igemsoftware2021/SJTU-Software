import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from utils.aligning import bp_align
from utils.embedding import OneHotEmbedding
from utils.label_embedding import *
from model import *
from Getdataset import IntelDNADataset
from _loss import *


writer = SummaryWriter('./Logs')

# Definition of Embedding Function
Embedder = OneHotEmbedding(4)
BinLabelEmbedder = BinLabelOneHotEmbedding(2)

# Hyper-Parameters
batch_size = 8
lr=0.001
global_length = 150
n_epoch = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definition of model utils
MODEL = Binary_TagModel_150(4, Block, 2)
MODEL = MODEL.to(device)
opti = torch.optim.Adam(MODEL.parameters(), lr=lr)
criterion = Tag_F1_loss

# Create dataset
trainSet = IntelDNADataset('./testset', 'train')
testSet = IntelDNADataset('./testset', 'test')
train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=False, drop_last=True)

train_step = 0
test_step = 0

for epoch in range(n_epoch):

    name = '100WModelState'

    print("Epoch{}".format(epoch))

    MODEL.train()

    for data in tqdm(train_dataloader):

        # Load data
        bp, label = data

        # Data aligning
        align_bp, align_label = bp_align(bp, label, batch_size, max_len=global_length)

        # One hot encoding
        encode_bp = [0 for x in range(batch_size)]
        encode_label = [0 for x in range(batch_size)]

        for i in range(batch_size):
            encode_bp[i] = Embedder.encode(align_bp[i])
            encode_label[i] = BinLabelEmbedder.encode(align_label[i])

        encode_bp = torch.tensor(encode_bp)
        encode_bp = encode_bp.permute(0, 2, 1)

        encode_label = torch.tensor(encode_label)
        encode_label = encode_label.permute(0, 2, 1)

        encode_bp = encode_bp.to(device)
        encode_label = encode_label.to(device)

        non_hot_label = torch.argmax(encode_label, dim=2)

        # Prediction
        out = MODEL(encode_bp)

        index_out = torch.argmax(out, dim=2)

        opti.zero_grad()

        loss = criterion(index_out, non_hot_label)

        loss.backward()

        opti.step()

        train_step += 1

        writer.add_scalar('Loss', loss, global_step=train_step)

    MODEL.eval()

    for data in tqdm(test_dataloader):

        # Load data
        bp, label = data

        # Data aligning
        align_bp, align_label = bp_align(bp, label, batch_size, max_len=global_length)

        # One hot encoding
        encode_bp = [0 for x in range(batch_size)]
        encode_label = [0 for x in range(batch_size)]

        for i in range(batch_size):
            encode_bp[i] = Embedder.encode(align_bp[i])
            encode_label[i] = BinLabelEmbedder.encode(align_label[i])

        encode_bp = torch.tensor(encode_bp)
        encode_bp = encode_bp.permute(0, 2, 1)

        encode_label = torch.tensor(encode_label)
        encode_label = encode_label.permute(0, 2, 1)

        encode_bp = encode_bp.to(device)
        encode_label = encode_label.to(device)

        non_hot_label = torch.argmax(encode_label, dim=2)

        with torch.no_grad():

            out = MODEL(encode_bp)

            index_out = torch.argmax(out, dim=2)

            test_loss = criterion(index_out, non_hot_label)

            test_step += 1

        writer.add_scalar('Test Loss', test_loss, global_step=test_step)

    name = name + str(epoch) + '.pth'
    torch.save(MODEL.state_dict(), name)


writer.close()
