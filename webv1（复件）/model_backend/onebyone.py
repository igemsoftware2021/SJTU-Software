import torch
import numpy as np
import torch.nn as nn
class BiLSTM(nn.Module):
    def __init__(self, input_dim, n_hidden):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, n_hidden, bidirectional=True)

    def forward(self, X):
        # X : [batch_size, seq_len, input_dim]
        input = X.transpose(0, 1)  # input : [seq_len, batch_size, embedding_dim]
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.transpose(0, 1)  # output : [batch_size, seq_len, n_hidden]
        return output

class Block2(nn.Module):
    def __init__(self, in_channel, drop_rate):
        super(Block2, self).__init__()
        self.cov1 = nn.Conv1d(in_channel, in_channel, 3, padding=1)
        self.cov2 = nn.Conv1d(in_channel, in_channel, 5, padding=2)
        self.Block2 = nn.Sequential(
            self.cov1,
            nn.BatchNorm1d(in_channel),
            nn.ELU(),
            nn.Dropout(drop_rate),
            self.cov2,
            nn.BatchNorm1d(in_channel),
            nn.ELU(),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        y = self.Block2(x)
        return x + y


class RNAModel_binary(nn.Module):
    def __init__(self, in_channel, layer_number,max_len,drop_rate=0.5):
        super(RNAModel_binary, self).__init__()
        self.max_len = max_len
        self.layer_number = layer_number

        layers_new = []
        for i in range(layer_number):
            al = Block2(in_channel=in_channel,drop_rate = drop_rate)
            setattr(self, "al%i" % i, al)
            layers_new.append(al)
        self.ResLayer_1d = nn.Sequential(
            *layers_new
        )

        self.Encoder = nn.TransformerEncoderLayer(in_channel, 2)
        self.Encoder_Layer = nn.TransformerEncoder(self.Encoder, num_layers=1)

        self.Bi_layer1 = BiLSTM(input_dim=in_channel,n_hidden=in_channel)
        self.Bi_layer2 = BiLSTM(input_dim=in_channel*2,n_hidden=in_channel*2)

        self.preditlayer1 = nn.Sequential(
            nn.Linear(16*self.max_len,
                      512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(drop_rate),
        )

        self.preditlayer2 = nn.Sequential(
            nn.Linear(512,
                      self.max_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.ResLayer_1d(x)
        x = x.permute(0, 2, 1)
        x = self.Encoder_Layer(x)
        x = self.Bi_layer1(x)
        x = self.Bi_layer2(x)
        x = x.reshape(x.shape[0],-1)
        x = self.preditlayer1(x)
        x = self.preditlayer2(x)
        return x.view(-1, self.max_len)


# 接受一个str, 返回一个L*3的标签
def dash2label(str,length):
    mat = torch.zeros((length,3))
    for i in range(len(str)):
        if(str[i]=='('):
            mat[i] = torch.tensor((1,0,0))
        elif(str[i] == '.'):
            mat[i] = torch.tensor((0,1,0))
        elif(str[i] == ')'):
            mat[i] = torch.tensor((0,0,1))
    return mat

def label2dash(out):
    dash = []
    for i in range(len(out)):
        if(out[i]==0):
            dash.append('(')
        elif(out[i]==1):
            dash.append('.')
        elif(out[i]==2):
            dash.append(')')
    return dash

# 接受一个str, 返回一个L*2的标签
def dash2label_binary(str,length):
    mat = torch.zeros((length,2))
    for i in range(len(str)):
        if(str[i]=='(' or str[i] == ')'):
            mat[i] = torch.tensor((1,0,))
        elif(str[i] == '.'):
            mat[i] = torch.tensor((0,1))
    return mat

# 接受一个str, 返回一个L*1的标签
def dash2label_1d(str,length):
    mat = torch.zeros(length)
    for i in range(len(str)):
        if(str[i]=='(' or str[i] == ')'):
            mat[i] = 1
        elif(str[i] == '.'):
            mat[i] = 0
    return mat

def out2label(tensor,length):
    mat = torch.zeros((tensor.shape[0],length))
    for k in range(tensor.shape[0]):
        for i in range(tensor.shape[1]):
            if (tensor[k][i] >= 0.5):
                mat[k][i] = 1
            elif (tensor[k][i] < 0.5):
                mat[k][i] = 0
    return mat

# align func
def align(seq, max_len=100):
    blank = '0'
    align_seq = seq + blank * (max_len - len(seq))
    return align_seq

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneHotEmbedding(nn.Module):
    def __init__(self, ksize=0):
        super(OneHotEmbedding, self).__init__()
        self.n_out = 4
        self.ksize = ksize
        eye = np.identity(4, dtype=np.float32)
        zero = np.zeros(4, dtype=np.float32)
        self.onehot = defaultdict(lambda: np.ones(4, dtype=np.float32)/4,
                {'a': eye[0], 'c': eye[1], 'g': eye[2], 't': eye[3], 'u': eye[3],
                    '0': zero})

    def encode(self, seq):
        seq = [ self.onehot[s] for s in seq.lower() ]
        seq = np.vstack(seq)
        return seq.transpose()

    def pad_all(self, seq, pad_size):
        pad = 'n' * pad_size
        seq = [ pad + s + pad for s in seq ]
        l = max([len(s) for s in seq])
        seq = [ s + '0' * (l-len(s)) for s in seq ]
        return seq

    def forward(self, seq):
        seq = self.pad_all(seq, self.ksize//2)
        seq = [ self.encode(s) for s in seq ]
        return torch.from_numpy(np.stack(seq)) # pylint: disable=no-member


class SparseEmbedding(nn.Module):
    def __init__(self, dim):
        super(SparseEmbedding, self).__init__()
        self.n_out = dim
        self.embedding = nn.Embedding(6, dim, padding_idx=0)
        self.vocb = defaultdict(lambda: 5,
            {'0': 0, 'a': 1, 'c': 2, 'g': 3, 't': 4, 'u': 4})


    def __call__(self, seq):
        seq = torch.LongTensor([[self.vocb[c] for c in s.lower()] for s in seq])
        seq = seq.to(self.embedding.weight.device)
        return self.embedding(seq).transpose(1, 2)

# good
# seq = 'GCUUCUAUGGCCAAGUUGGUAAGGCGCCACACUAGUAAUGUGGAGAUCAUCGGUUCAAAUCCGAUUGGAAGCACCA'
# structure = '(((((((..(((..........))).(((((.......))))).....(((((.......))))))))))))....'

# good good
seq = 'CGGGAUGUGGCCCAGCUUGGUAGGGCACUGCGUUCGGGACGCAGGAGUCGCGCGUUCAAAUCGCGCCAUCCCGACCA'
structure = '(((((((..((((.........))))((((((.......))))))....(((((.......))))))))))))....'

# bad
# seq = 'CAAGUCGUCAAUUUGGUUGUGAACUCUUAAUUUAAACGAUAUCACAGCCACUUUGAUGAGCUUGG'
# structure = '....((((((((((((((((((..(((...........)))))))))))))))))))))......'

# bad bad
# seq = 'GGGGAGCAGGGUCCGGCUUGGUUUGUACUUGGAUGGGAGACCGCCUGGGAAUACCAGGUGCUGUAAGCCUUUUC'
# structure = '................(((((......(((((.(((....)))))))).....)))))................'

# seq = 'CGCAGAGUAGAUAAGAUGCGUUAAGUGUUAAAGGAUGGGAAGUUGCCUUUAAACGAAAACGAAAGUUGCGGUAUCUUAUUGCGUCCGCUGUG'
# structure = '.((((.....(((((((((......(.(((((((....[[[[[...))))))))....((.....))...)))))))))).]]]]].)))..'

def get_result(seq):
    # Hyper-Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_length = 100
    in_channel = 4
    if_load = True
    init_param = "/home/egotist/SJTU-Software/webv1/model_backend/Model_binary_reinforce80.pth"

    MODEL = RNAModel_binary(in_channel=4, layer_number=2, max_len=global_length)

    # Definition of Embedding Function
    Embedder = OneHotEmbedding(4)

    # if_load_params
    if (if_load):
        MODEL.load_state_dict(torch.load(init_param))

    MODEL.eval()

    align_bp = align(seq)

    # One hot encoding
    encode_bp = Embedder.encode(align_bp)
    encode_bp = torch.tensor(encode_bp)
    encode_bp = encode_bp.view(-1, in_channel, 100)
    encode_bp = encode_bp.permute(0, 2, 1)

    #print(type(encode_bp), encode_bp.shape)

    # Prediction
    with torch.no_grad():
        out = MODEL(encode_bp)

    pred = out2label(out, global_length)
    label = []
    for i in range(len(seq)):
        if structure[i] == '.':
            label.append(0)
        else:
            label.append(1)

    pred = pred.squeeze().numpy()
    pred = pred.astype(dtype=int)

    return pred


# pred_result = get_result(seq)
# print(pred_result)