import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from model import *
from utils.aligning import bp_align
from utils.dash2mat import dash2matrix
from utils.embedding import *
from utils.label_embedding import *
from torch.utils.data import DataLoader
from Getdataset import IntelDNADataset
from _loss import *




param = torch.load('./8BatchModel1.pth')

batch_size = 1

global_length = 100

Embedder = OneHotEmbedding(4)
LabelEmbedder  = BinLabelOneHotEmbedding(2)

model = Binary_TagModel_100(4, Block, 2).to(device)

model.load_state_dict(param)

devset = IntelDNADataset('./dataset_50100', 'train')

dev_dataloader = DataLoader(devset, batch_size=batch_size, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_acc = torch.zeros(1).to(device)

for data in tqdm(dev_dataloader):

    bp, label = data

    # Data aligning
    align_bp, align_label = bp_align(bp, label, batch_size, max_len=global_length)

    # One hot encoding
    encode_bp = [0 for x in range(batch_size)]
    encode_label = [0 for x in range(batch_size)]

    for i in range(batch_size):
        encode_bp[i] = Embedder.encode(align_bp[i])
        encode_label[i] = LabelEmbedder.encode(align_label[i])

    encode_bp = torch.tensor(encode_bp)
    encode_bp = encode_bp.permute(0, 2, 1)

    encode_label = torch.tensor(encode_label)
    encode_label = encode_label.permute(0, 2, 1)

    encode_bp = encode_bp.to(device)
    encode_label = encode_label.to(device)

    non_hot_label = torch.argmax(encode_label, dim=2)

    with torch.no_grad():

        out = model(encode_bp.to(device))

        index_out = torch.argmax(out, dim=2)

        acc = (index_out == non_hot_label).float().mean()

        print(acc)
        # total_f1 = torch.zeros(1, requires_grad=True).to(device)
        # for i in range(index_out.shape[0]):
        #     f1 = torch.from_numpy(np.array(f1_score(index_out[i].detach().cpu(), non_hot_label[i].detach().cpu(), average='weighted')))
        #     f1 = f1.to(device)
        #     total_f1 = total_f1 + f1
        # loss = Tag_F1_loss(index_out, non_hot_label)

        # print()
        #
        # mean_acc = acc.sum(dim=0)/batch_size
        #
        # total_acc += mean_acc
        #
        # print(mean_acc)


