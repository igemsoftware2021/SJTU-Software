# import essential package
from torch.utils.data import DataLoader
from Getdataset import IntelDNADataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
# utils
from utils.aligning import bp_align
from utils.embedding import OneHotEmbedding
from utils.label_embedding import *
from utils.padding_mask import *
# loss function
from _loss import *
# model
from model.attention_model import *

# Hyper-Parameters
batch_size = 256
lr = 0.01
global_length = 100
n_epoch = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Experiment_name = 'Model_E13'

# Create tensorboard
writer = SummaryWriter(Experiment_name)

# Definition of Embedding Function
Embedder = OneHotEmbedding(4)
LabelEmbedder = LabelOneHotEmbedding(3)
position = PositionalEncoding(8, global_length)

# Definition of model
AttentionLayer = MODELE(8, 2, 1, 1, batch_size, 0.2, global_length)
AttentionLayer = AttentionLayer.to(device)
criterion = TagLoss
# Load model with dict
param_load_model = torch.load("./Model_B10.8858984112739563.pth", map_location='cuda')
AttentionLayer.load_state_dict(param_load_model)

# Optimizer & Loss Function
opti = torch.optim.Adam(AttentionLayer.parameters(), lr=lr)
# Create dataset
trainSet = IntelDNADataset('./data/data_v2', './data/adj_dataset', './data/vec_dataset', global_length, 'train')
testSet = IntelDNADataset('./data/data_v2', './data/adj_dataset', './data/vec_dataset', global_length, 'test')
train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=False, drop_last=True)

# Step to record
train_step = 0
test_step = 0

# minimum  Accuracy
min_acc = torch.tensor(0.85).to(device)

# Create base dict
src_vocab = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'U': 1, '#': 4, 'S': 5, 'N': 6, 'R': 7, 'D': 8, 'Y': 9}

# Main Epoch
for epoch in range(n_epoch):

    # Definition of Model Name
    name = Experiment_name

    print("Epoch{}".format(epoch))

    # Training Mode
    AttentionLayer.train()

    for data in tqdm(train_dataloader):

        # Accuracy
        train_acc = 0

        # Load data
        bp, label, adj, vec = data

        # Data aligning
        align_bp, align_label = bp_align(bp, label, batch_size, max_len=global_length)

        # Create Adj_matrix
        for i in range(batch_size):
            adj[i] = buildAdjandIdxmap(adj[i], global_length)

        adj = adj.detach().cpu().numpy()
        adj = torch.from_numpy(adj)
        adj = adj.to(device)

        # Data encode for src_padding_mask
        enc_input = []
        for batch in range(batch_size):
            enc_input.append([])
            for i in range(len(align_bp[batch])):
                enc_input[batch].append([src_vocab[n] for n in align_bp[batch][i].split()])

        enc_input = torch.tensor(enc_input)
        enc_input = enc_input.to(device)
        enc_input = enc_input.view(batch_size, -1)

        # Src padding mask
        src_mask = padding_mask(enc_input, global_length)

        # One hot Encoding
        encode_bp = [0 for x in range(batch_size)]
        encode_label = [0 for x in range(batch_size)]
        for i in range(batch_size):
            encode_bp[i] = Embedder.encode(align_bp[i])
            encode_label[i] = LabelEmbedder.encode(align_label[i])
        encode_bp = torch.tensor(encode_bp)
        encode_bp = encode_bp.permute(2, 0, 1)

        encode_label = torch.tensor(encode_label)
        encode_label = encode_label.permute(0, 2, 1)

        encode_label_index = torch.argmax(encode_label, dim=2)
        encode_label_index = encode_label_index.to(device)

        encode_bp = encode_bp.to(device)
        encode_label = encode_label.to(device)
        encode_label = encode_label.long()  # That's important, or IDE will deliver error

        vec = vec.permute(1 ,0, 2).to(device)

        pos_input = position(torch.cat((encode_bp, vec), dim=2)).float()

        # Prediction
        out = AttentionLayer(pos_input, adj, src_mask)

        index_out = torch.argmax(out, dim=2)

        index_out = index_out.to(device)

        opti.zero_grad()

        loss = criterion(out, encode_label_index)

        acc = (index_out == encode_label_index).float().mean()

        train_acc += acc

        loss.backward()

        opti.step()

        train_step += 1

        writer.add_scalar('Loss', loss, global_step=train_step)
        writer.add_scalar('Train Acc', train_acc, global_step=train_step)

    AttentionLayer.eval()

    for data in tqdm(test_dataloader):

        test_acc = 0

        acc_step = 0

        # Load data
        bp, label, adj, vec = data

        # Data aligning
        align_bp, align_label = bp_align(bp, label, batch_size, max_len=global_length)

        # Create Adj_matrix
        for i in range(batch_size):
            adj[i] = buildAdjandIdxmap(adj[i], global_length)

        adj = adj.detach().cpu().numpy()
        adj = torch.from_numpy(adj)
        adj = adj.to(device)

        # Data encode for src_padding_mask
        enc_input = []
        for batch in range(batch_size):
            enc_input.append([])
            for i in range(len(align_bp[batch])):
                enc_input[batch].append([src_vocab[n] for n in align_bp[batch][i].split()])

        enc_input = torch.tensor(enc_input)
        enc_input = enc_input.to(device)
        enc_input = enc_input.view(batch_size, -1)

        # Src padding mask
        src_mask = padding_mask(enc_input, global_length)

        # One hot Encoding
        encode_bp = [0 for x in range(batch_size)]
        encode_label = [0 for x in range(batch_size)]
        for i in range(batch_size):
            encode_bp[i] = Embedder.encode(align_bp[i])
            encode_label[i] = LabelEmbedder.encode(align_label[i])
        encode_bp = torch.tensor(encode_bp)
        encode_bp = encode_bp.permute(2, 0, 1)

        encode_label = torch.tensor(encode_label)
        encode_label = encode_label.permute(0, 2, 1)

        encode_label_index = torch.argmax(encode_label, dim=2)
        encode_label_index = encode_label_index.to(device)

        encode_bp = encode_bp.to(device)
        encode_label = encode_label.to(device)
        encode_label = encode_label.long()  # That's important, or IDE will deliver error

        vec = vec.permute(1, 0, 2).to(device)

        pos_input = position(torch.cat((encode_bp, vec), dim=2)).float()

    with torch.no_grad():

        out = AttentionLayer(pos_input, adj, src_mask)

        index_out = torch.argmax(out, dim=2)

        index_out = index_out.to(device)

        test_loss = criterion(out, encode_label_index)

        acc = (index_out == encode_label_index).float().mean()

        test_acc += acc

        test_step += 1

        writer.add_scalar('Test Loss', test_loss, global_step=test_step)
        writer.add_scalar('Test Acc', test_acc, global_step=test_step)

    if test_acc > min_acc:
        min_acc = test_acc
        name = name + str(test_acc.item()) + '.pth'
        torch.save(AttentionLayer.state_dict(), name)

writer.close()
