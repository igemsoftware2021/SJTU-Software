from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

class LabelOneHotEmbedding(nn.Module):
    def __init__(self, ksize=0):
        super(LabelOneHotEmbedding, self).__init__()
        self.n_out = 3
        self.ksize = ksize
        eye = np.identity(3, dtype=np.float32)
        zero = np.zeros(3, dtype=np.float32)
        mask = np.array([float('-inf') for x in range(self.n_out)], dtype=np.float32)
        self.onehot = defaultdict(lambda: np.ones(3, dtype=np.float32)/3,
                {'.': eye[0], '(': eye[1], ')': eye[2], '0': zero, '#': zero})

    def encode(self, seq):
        seq = [ self.onehot[s] for s in seq ]
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


class BinLabelOneHotEmbedding(nn.Module):
    def __init__(self, ksize=0):
        super(BinLabelOneHotEmbedding, self).__init__()
        self.n_out = 2
        self.ksize = ksize
        eye = np.identity(2, dtype=np.float32)
        zero = np.zeros(2, dtype=np.float32)
        self.onehot = defaultdict(lambda: np.ones(2, dtype=np.float32)/2,
                {'.': eye[0], '(': eye[1], ')': eye[1], '0': zero})

    def encode(self, seq):
        seq = [ self.onehot[s] for s in seq ]
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