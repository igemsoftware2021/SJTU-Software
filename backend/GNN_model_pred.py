from torch.utils.data import DataLoader
from Getdataset import IntelDNADataset

# utils
from utils.aligning import bp_align, de_align
from utils.embedding import OneHotEmbedding
from utils.mask import *
from utils.get_adj import *
from utils.myNode2vec import node2vector
import matplotlib.pyplot as plt

# model
from model.attention_model import *
from sklearn.metrics import f1_score, precision_score
import warnings
warnings.filterwarnings("ignore")

# Hyper-Parameters
batch_size = 1
global_length = 100
n_epoch = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definition of Embedding Function
Embedder = OneHotEmbedding(4)
LabelEmbedder = OneHotEmbedding(2)
position = PositionalEncoding(8, global_length)

src_vocab = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'U': 1, '#': 4, 'R': 4, 'Y': 4, 'M': 4, 'K': 4, 'S': 4, 'W': 4, 'B': 4,
             'V': 4, 'D': 4, 'N': 4, 'I': 4}

def get_pred_GNN_binary(seq):

    # Definition of model
    model_bs = 256
    AttentionLayer = MODELC(8, 1, 1, 1, model_bs, 0.25, global_length)
    AttentionLayer = AttentionLayer.to(device)
    # Load model with dict
    param_load_model = torch.load("./GNN_Model/Model_C10.88148432970047.pth", map_location='cpu')
    AttentionLayer.load_state_dict(param_load_model)

    AttentionLayer.eval()

    length = len(seq)
    # Data aligning
    bp = [x.upper() for x in seq]
    align_bp, _ = bp_align(bp, "aaa", batch_size, max_len=global_length)

    # get adj list (actually only one)
    adj_mat = getadj(seq,100)
    adj = torch.zeros((1,100,100))
    adj[0] = adj_mat

    vec = node2vector(adj[0].numpy(), 100, 4)
    vec_zero = torch.zeros([1, 100, 4])
    vec_zero[0] = torch.from_numpy(vec)
    vec = vec_zero.permute(1, 0, 2).to(device)

    # Create Adj_matrix
    for i in range(batch_size):
        adj[i] = buildAdjandIdxmap(adj[i], global_length)
    adj = adj.detach().cpu().numpy()
    adj = torch.from_numpy(adj)
    adj = adj.to(device)
    # print(adj.shape) [1,100,100]

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
    for i in range(batch_size):
        encode_bp[i] = Embedder.encode(align_bp[i])
    encode_bp = torch.tensor(encode_bp)
    encode_bp = encode_bp.permute(2, 0, 1)

    encode_bp = encode_bp.to(device)
    # print(encode_bp.shape) [100,1,4]

    # L, B, d
    pos_input = position(torch.cat((encode_bp, vec), dim=2)).float()

    pos_input = pos_input.repeat(1, model_bs, 1)

    src_mask = src_mask.repeat(model_bs, 1)

    adj = adj.repeat(model_bs, 1, 1)

    def class2binnota(x):
        if x == 0:
            return '.'
        if x == 1:
            return '@'

    with torch.no_grad():
        # Prediction
        out = AttentionLayer(pos_input, adj, src_mask)
        seq = de_align(out, batch_size, length)
        index_out = torch.argmax(seq[0], dim=1)
        index_out = index_out.to(device)
        index_out = index_out.detach().cpu().tolist()
        pred = [class2binnota(seq) for seq in index_out]
    return ''.join(pred)

def get_pred_GNN_tri(seq):
    # 还需要vec!!!!!!!!

    # Definition of model
    model_bs = 256
    AttentionLayer = MODELE(8, 2, 1, 1, model_bs, 0.2, global_length)
    AttentionLayer = AttentionLayer.to(device)
    AttentionLayer.eval()

    # Load model with dict
    param_load_model = torch.load("./GNN_Model/Model_E130.8878124952316284.pth", map_location='cpu')
    AttentionLayer.load_state_dict(param_load_model)

    length = len(seq)
    # Data aligning
    bp = [x.upper() for x in seq]
    align_bp, _ = bp_align(bp, "aaa", batch_size, max_len=global_length)

    # get adj list (actually only one)
    adj_mat = getadj(seq,100)
    adj = torch.zeros((1,100,100))
    adj[0] = adj_mat

    vec = node2vector(adj[0].numpy(), 100, 4)
    vec_zero = torch.zeros([1, 100, 4])
    vec_zero[0] = torch.from_numpy(vec)
    vec = vec_zero.permute(1, 0, 2).to(device)

    # Create Adj_matrix
    for i in range(batch_size):
        adj[i] = buildAdjandIdxmap(adj[i], global_length)
    adj = adj.detach().cpu().numpy()
    adj = torch.from_numpy(adj)
    adj = adj.to(device)
    # print(adj.shape) [1,100,100]

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
    for i in range(batch_size):
        encode_bp[i] = Embedder.encode(align_bp[i])
    encode_bp = torch.tensor(encode_bp)
    encode_bp = encode_bp.permute(2, 0, 1)

    encode_bp = encode_bp.to(device)
    # print(encode_bp.shape) [100,1,4]

    # L, B, d
    pos_input = position(torch.cat((encode_bp, vec), dim=2)).float()

    pos_input = pos_input.repeat(1, model_bs, 1)

    src_mask = src_mask.repeat(model_bs, 1)

    adj = adj.repeat(model_bs, 1, 1)

    def class2nota(x):
        if x == 0:
            return '.'
        if x == 1:
            return '('
        if x == 2:
            return ')'

    with torch.no_grad():
        # Prediction
        out = AttentionLayer(pos_input, adj, src_mask)
        seq = de_align(out, batch_size, length)
        index_out = torch.argmax(seq[0], dim=1)
        index_out = index_out.to(device)
        index_out = index_out.detach().cpu().tolist()
        pred = [class2nota(seq) for seq in index_out]
    return ''.join(pred)

def get_struct_score(seq,pred):
    score = 0
    return score

if(__name__ == '__main__'):
    # good good
    seq = 'CGGGAUGUGGCCCAGCUUGGUAGGGCACUGCGUUCGGGACGCAGGAGUCGCGCGUUCAAAUCGCGCCAUCCCGACCA'
    structure = '(((((((..((((.........))))((((((.......))))))....(((((.......))))))))))))....'
    # mat = getadj(seq,100)

    seq2 = 'ATCTATCCACCTCCACCTCTACACCCATGAGATGAGTTGGCAATGGTAGAACT'

    pred_bi = get_pred_GNN_binary(seq2)
    pred_tri = get_pred_GNN_tri(seq2)
    print(pred_bi)
    print(pred_tri)

    # seq3 = "acacgaggaggttttaca"
    #
    # pred3_bi = get_pred_GNN_tri(seq3)
    # print(pred3_bi)

