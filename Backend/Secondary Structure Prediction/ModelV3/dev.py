# import essential package
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score
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
import warnings

warnings.filterwarnings("ignore")

# Hyper-Parameters
batch_size = 1
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
model_bs = 256
AttentionLayer = MODELO(wv_size, 4, 1, 2, model_bs, 0.25, global_length)
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
devSet = IntelDNADataset('./dataset/bpRNA100', './dataset/bpRNA100_ADJ', global_length, wv_size, 'test')
dev_dataloader = DataLoader(devSet, batch_size=batch_size, shuffle=False, drop_last=True)

# minimum  Accuracy
min_acc = torch.tensor(0.85).to(device)

# Create base dict notation "4" represent for padding mask
src_vocab = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'U': 1, '#': 4}

total_acc = []
total_f1 = []
total_presion = []
total_rog = []
cohen_score = []

# Main Epoch
for epoch in range(n_epoch):

    # Definition of Model Name
    name = Experiment_name

    print("Epoch{}".format(epoch))

    # Training Mode
    AttentionLayer.train()

    for data in tqdm(dev_dataloader):

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

        bp_input = bp_input.repeat(1, model_bs, 1)

        src_mask = src_mask.repeat(model_bs, 1)

        adj = adj.repeat(model_bs, 1, 1)

        # Prediction
        with torch.no_grad():

            out = AttentionLayer(bp_input, adj, src_mask)

        index_out = torch.argmax(seq[0], dim=1)

        index_out = index_out.to(device)

        # target_names = ['class 0', 'class 1', 'class 2']
        #
        # print(classification_report(encode_label_index[0].detach().cpu().tolist(), index_out[0].detach().cpu().tolist(), target_names=target_names))

        f1 = f1_score(encode_label_index[0].detach().cpu().tolist(), index_out.detach().cpu().tolist(),
                      average='micro')
        #
        precison = precision_score(encode_label_index[0].detach().cpu().tolist(),
                                   index_out.detach().cpu().tolist(), average='micro')

        # rog = roc_auc_score(encode_label_index[0].detach().cpu().tolist(), index_out[0].detach().cpu().tolist(), average='weighted', multi_class='ovr')

        acc = (index_out == encode_label_index).float().mean()

        acc = acc.detach().cpu().tolist()

        index_out = index_out.detach().cpu().tolist()

        # cohen_score.append(
        #     cohen_kappa_score(encode_label_index[0].detach().cpu().tolist(), index_out[0].detach().cpu().tolist()))

        if acc < min_acc:
            min_acc = acc

        total_acc.append(acc)
        total_f1.append(f1)
        total_presion.append(precison)
        # total_rog.append(rog)

        print('Acc:{}'.format(acc))
        print('Label {}'.format(label[0]))
        print('F1:{}'.format(f1))
        print('Presion:{}'.format(precison))




def list2average(list):
    sum = 0

    for item in list:
        sum += item

    return sum / len(list)

def class2nota(x):
    if x == 0:
        return '.'
    if x == 1:
        return '('
    if x == 2:
        return ')'

def class2binnota(x):
    if x == 0:
        return '.'
    if x == 1:
        return '@'


    print('Min_acc: {}'.format(min_acc))
    print('Total Acc:{}'.format(list2average(total_acc)))
    print('Total F1:{}'.format(list2average(total_f1)))
    print('Total Presion:{}'.format(list2average(total_presion)))
    # print('Total Rog:{}'.format(list2average(total_rog)))

    plt.title('Acc')
    plt.boxplot(total_acc)
    plt.show()

    # plt.title('Precision')
    # plt.boxplot(total_presion)
    # plt.show()
    #
    plt.title('F1')
    plt.boxplot(total_f1)
    plt.show()
