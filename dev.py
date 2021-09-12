# import essential package
from torch.utils.data import DataLoader
from Getdataset import IntelDNADataset

# utils
from utils.aligning import bp_align
from utils.embedding import OneHotEmbedding
from utils.label_embedding import *
from utils.padding_mask import *
import matplotlib.pyplot as plt

# model
from model.attention_model import *
from sklearn.metrics import f1_score, precision_score, cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")

# Hyper-Parameters
batch_size = 1
lr = 0.0005
global_length = 100
n_epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Definition of Embedding Function
Embedder = OneHotEmbedding(4)
LabelEmbedder = LabelOneHotEmbedding(3)
position = PositionalEncoding(8, global_length)

# Definition of model
model_bs = 256
AttentionLayer = MODELE(8, 1, 1, 1, model_bs, 0.25, global_length)
AttentionLayer = AttentionLayer.to(device)
# Load model with dict
param_load_model = torch.load("./Model_E10.8744140267372131.pth", map_location='cuda')
AttentionLayer.load_state_dict(param_load_model)

# Create dataset
devSet = IntelDNADataset('./data/data_v2', './data/adj_dataset', './data/vec_dataset', global_length, 'dev')
dev_dataloader = DataLoader(devSet, batch_size=batch_size, shuffle=True, drop_last=True)

# Step to record
dev_step = 0

# Create base dict
src_vocab = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'U': 1, '#': 4, 'S': 5, 'N': 6, 'R': 7, 'D': 8, 'Y': 9}

total_acc = []
min_acc = 0
total_f1 = []
total_presion = []
total_rog = []
cohen_score = []

def list2average(list):

    sum = 0

    for item in list:

        sum += item

    return sum/len(list)

# Main Epoch
for epoch in range(n_epoch):

    # Definition of Model Name

    # De Mode
    AttentionLayer.eval()

    for data in dev_dataloader:

        # Accuracy
        dev_acc = 0

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

        # L, B, d
        pos_input = position(torch.cat((encode_bp, vec), dim=2)).float()

        pos_input = pos_input.repeat(1, model_bs, 1)

        src_mask = src_mask.repeat(model_bs, 1)

        adj = adj.repeat(model_bs, 1, 1)

        with torch.no_grad():
            # Prediction
            out = AttentionLayer(pos_input, adj, src_mask)

            index_out = torch.argmax(out, dim=2)

            index_out = index_out.to(device)

            # target_names = ['class 0', 'class 1', 'class 2']

            # print(classification_report(encode_label_index[0].detach().cpu().tolist(), index_out[0].detach().cpu().tolist(), target_names=target_names))

            f1 = f1_score(encode_label_index[0].detach().cpu().tolist(), index_out[0].detach().cpu().tolist(), average='weighted')

            precison = precision_score(encode_label_index[0].detach().cpu().tolist(), index_out[0].detach().cpu().tolist(), average='weighted')

            # rog = roc_auc_score(encode_label_index[0].detach().cpu().tolist(), index_out[0].detach().cpu().tolist(), average='weighted', multi_class='ovr')

            acc = (index_out == encode_label_index).float().mean()

            acc = acc.detach().cpu().tolist()

            cohen_score.append(cohen_kappa_score(encode_label_index[0].detach().cpu().tolist(), index_out[0].detach().cpu().tolist()))

            if acc < min_acc:
                min_acc = acc

            total_acc.append(acc)
            total_f1.append(f1)
            # total_rog.append(rog)
            total_presion.append(precison)

            # print('Acc:{}'.format(acc))
            # print('F1:{}'.format(f1))
            # print('Presion:{}'.format(precison))
            # print('Rog:{}'.format(rog))

    print('Min_acc: {}'.format(min_acc))
    print('Total Acc:{}'.format(list2average(total_acc)))
    print('Total F1:{}'.format(list2average(total_f1)))
    print('Total Presion:{}'.format(list2average(total_presion)))
    print('cohen_scores:{}'.format(list2average(cohen_score)))
    # print('Total Rog:{}'.format(list2average(total_rog)))

    plt.title('Acc')
    plt.boxplot(total_acc)
    plt.show()

    plt.title('Precision')
    plt.boxplot(total_presion)
    plt.show()

    plt.title('F1')
    plt.boxplot(total_f1)
    plt.show()

    plt.title('cohen_scores')
    plt.boxplot(cohen_score)
    plt.show()

