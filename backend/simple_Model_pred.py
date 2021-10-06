import numpy as np
import torch
from utils.embedding import OneHotEmbedding
from model_1d import *
from utils.dash2label import *
from _loss import *
import numpy as np

def align(seq, max_len=100):
    blank = '0'
    align_seq = seq + blank * (max_len - len(seq))
    return align_seq

def get_simple_pred(seq):
    # Hyper-Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_length = 100
    in_channel = 4
    if_load = True
    init_param = "./simple_Model/Model_binary_reinforce80.pth"

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
    pred = pred[:len(seq)].tolist()
    pred = [str(i) for i in pred]

    return ''.join(pred)

if(__name__ == '__main__'):
    seq = 'CGGGAUGUGGCCCAGCUUGGUAGGGCACUGCGUUCGGGACGCAGGAGUCGCGCGUUCAAAUCGCGCCAUCCCGACCA'
    structure = '(((((((..((((.........))))((((((.......))))))....(((((.......))))))))))))....'

    seq2 = 'GCUGUCACCGGAAUAACCGAAGUAGUUUAUGCGCUACCGAAAUCCGGUCUGAUGAGUCCUCGGGGUGGACGAAACAGC'

    pred = get_simple_pred(seq)

    pred2 = get_simple_pred(seq2)
    print(structure)
    print(pred)
    print(pred2)