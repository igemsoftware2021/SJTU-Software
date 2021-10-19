# import essential package
import torch
from torch.utils.data import DataLoader
from utils.Getdataset import IntelDNADataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
# utils
from utils.aligning import *
from utils.embedding import *
from utils.label_embedding import *
from utils.mask import *
# loss function
from utils._loss import *
# model
from model.attention_model import *
from utils.calculate import model_structure

# Hyper-Parameters
batch_size = 256
wv_size = 16
lr = 0.0005
global_length = 100
n_epoch = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Experiment_name = 'ModelV2'

# Create tensorboard
writer = SummaryWriter(Experiment_name)

# Definition of Embedding Function
LabelEmbedder = LabelOneHotEmbedding(3)
position = PositionalEncoding(wv_size, global_length)

# Definition of model
torch.cuda.empty_cache()
AttentionLayer = MODELO(wv_size, 4, 1, 2, batch_size, 0.25, global_length)
AttentionLayer = AttentionLayer.to(device)
criterion_bce_weighted = torch.nn.BCEWithLogitsLoss()
criterion = f1_loss
# Load model with dict
param_load_model = torch.load("./ModelV2_Stage0.pth", map_location='cuda')
AttentionLayer.load_state_dict(param_load_model)


# calculate parameters
model_structure(AttentionLayer)

# Optimizer & Loss Function
opti = torch.optim.Adam(AttentionLayer.parameters(), lr=lr)
# Create dataset
trainSet = IntelDNADataset('./dataset/bpRNA100', './dataset/bpRNA100_ADJ', global_length, wv_size, 'train')
testSet = IntelDNADataset('./dataset/bpRNA100', './dataset/bpRNA100_ADJ', global_length, wv_size, 'test')
train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=False, drop_last=True)

# Step to record
train_step = 0
test_step = 0

# minimum  Accuracy
min_acc = torch.tensor(0.85).to(device)

# Create base dict notation "4" represent for padding mask
src_vocab = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'U': 1, '#': 4}

# Main Epoch
for epoch in range(n_epoch):

    # Definition of Model Name
    name = Experiment_name

    print("Epoch{}".format(epoch))

    # Training Mode
    AttentionLayer.train()

    for data in tqdm(train_dataloader):

        # Load data
        # seq, encode_bp, label, adj, Codes, lens = data

        seq, wv_align_bp, label, adj, lens = data


        # wv_align_bp = encodecp_align(encode_bp, batch_size, wv_size, global_length)
        align_label = target_align(label, batch_size, global_length)

        # Data encode for src_padding_mask
        enc_input = []
        for batch in range(batch_size):
            enc_input.append([])
            for i in range(len(seq[batch])):
                enc_input[batch].append([src_vocab[n] for n in seq[batch][i].split()])
        enc_input = torch.tensor(enc_input)
        enc_input = enc_input.to(device)
        enc_input = enc_input.view(batch_size, -1)

        # Src padding mask
        src_mask = padding_mask(enc_input, global_length)

        # One hot Encoding
        encode_bp = [0 for x in range(batch_size)]
        encode_label = [0 for x in range(batch_size)]
        for i in range(batch_size):
            encode_label[i] = LabelEmbedder.encode(align_label[i])

        encode_bp = torch.tensor(wv_align_bp)
        encode_bp = encode_bp.permute(1, 0, 2)

        encode_label = torch.tensor(encode_label)
        encode_label = encode_label.permute(0, 2, 1)

        encode_label_index = torch.argmax(encode_label, dim=2)
        encode_label_index = encode_label_index.to(device)

        encode_bp = encode_bp.to(device)
        encode_label = encode_label.to(device)
        adj = adj.to(device)

        bp_input = position(encode_bp)
        bp_input = bp_input.permute(1, 0, 2)

        ## encode_label = encode_label.long()  # That's important, or IDE will deliver error

        # contact_masks = torch.Tensor(contact_map_masks(lens, global_length)).to(device)
        #
        # contacts = [0 for x in range(batch_size)]
        #
        # for i in range(batch_size):
        #     contacts[i] = dash2matrix(align_label[i], global_length)
        #
        # contacts = torch.tensor([item.cpu().detach().numpy() for item in contacts])
        #
        # contacts_batch = contacts.float().to(device)

        # Prediction
        out = AttentionLayer(bp_input, adj, src_mask)

        opti.zero_grad()

        loss = criterion(out, encode_label)

        index_out = torch.argmax(out, dim=2).to(device)

        acc = (index_out == encode_label_index).float().mean()

        torch.cuda.empty_cache()

        loss.backward()

        opti.step()

        train_step += 1

        writer.add_scalar('Loss', loss, global_step=train_step)
        writer.add_scalar('Acc', acc, global_step=train_step)

    AttentionLayer.eval()

    for data in tqdm(test_dataloader):

        seq, wv_align_bp, label, adj, lens = data

        align_bp = bp_align(seq, batch_size, global_length)
        # wv_align_bp = encodecp_align(encode_bp, batch_size, wv_size, global_length)
        align_label = target_align(label, batch_size, global_length)

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
            encode_label[i] = LabelEmbedder.encode(align_label[i])

        encode_bp = torch.tensor(wv_align_bp)
        encode_bp = encode_bp.permute(1, 0, 2)

        encode_label = torch.tensor(encode_label)
        encode_label = encode_label.permute(0, 2, 1)

        encode_label_index = torch.argmax(encode_label, dim=2)
        encode_label_index = encode_label_index.to(device)

        encode_bp = encode_bp.to(device)
        encode_label = encode_label.to(device)
        adj = adj.to(device)

        bp_input = position(encode_bp)
        bp_input = bp_input.permute(1, 0, 2)
        ## encode_label = encode_label.long()  # That's important, or IDE will deliver error

        # contact_masks = torch.Tensor(contact_map_masks(lens, global_length)).to(device)
        #
        # contacts = [0 for x in range(batch_size)]
        #
        # for i in range(batch_size):
        #     contacts[i] = dash2matrix(align_label[i], global_length)
        #
        # contacts = torch.tensor([item.cpu().detach().numpy() for item in contacts])
        #
        # contacts_batch = contacts.float().to(device)
    with torch.no_grad():

        # Prediction

        out = AttentionLayer(bp_input, adj, src_mask)

        loss = criterion(out, encode_label)

        index_out = torch.argmax(out, dim=2).to(device)

        acc = (index_out == encode_label_index).float().mean()

        torch.cuda.empty_cache()

        test_step += 1

        writer.add_scalar('Test Loss', loss, global_step=test_step)
        writer.add_scalar('Test Acc', acc, global_step=test_step)

    if acc > min_acc:
        min_acc = acc
        name = name + str(acc.item()) + '.pth'
        torch.save(AttentionLayer.state_dict(), name)

writer.close()
