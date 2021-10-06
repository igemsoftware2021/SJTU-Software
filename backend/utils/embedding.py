from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
import torch.nn.functional as F


class OneHotEmbedding(nn.Module):
    def __init__(self, ksize=0):
        super(OneHotEmbedding, self).__init__()
        self.n_out = 4
        self.ksize = ksize
        eye = np.identity(4, dtype=np.float32)
        zero = np.zeros(4, dtype=np.float32)
        mask = np.array([float('-inf') for x in range(self.n_out)], dtype=np.float32)
        self.onehot = defaultdict(lambda: np.ones(4, dtype=np.float32) / 4,
                                  {'a': eye[0], 'c': eye[1], 'g': eye[2], 't': eye[3], 'u': eye[3],
                                   'R': zero, '#': zero, 'Y': zero, 'M': zero, 'K': zero, 'S': zero, 'W': zero,
                                   'B': zero, 'V': zero, 'D': zero, 'N': zero})

    def encode(self, seq):
        seq = [self.onehot[s] for s in seq.lower()]
        seq = np.vstack(seq)
        return seq.transpose()

    def pad_all(self, seq, pad_size):
        pad = 'n' * pad_size
        seq = [pad + s + pad for s in seq]
        l = max([len(s) for s in seq])
        seq = [s + '0' * (l - len(s)) for s in seq]
        return seq

    def forward(self, seq):
        seq = self.pad_all(seq, self.ksize // 2)
        seq = [self.encode(s) for s in seq]
        return torch.from_numpy(np.stack(seq))  # pylint: disable=no-member


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

class WVEmbedding(nn.Module):
    def __init__(self, ksize=0):
        super(WVEmbedding, self).__init__()
        model = Word2Vec.load('./utils/Word.ckpt')
        self.n_out = 16
        self.ksize = ksize
        Code_A = model.__getitem__("A")
        Code_U = model.__getitem__("U")
        Code_C = model.__getitem__("C")
        Code_G = model.__getitem__("G")
        zero = np.zeros(self.ksize, dtype=np.float32)
        mask = np.array([float('-inf') for x in range(self.n_out)], dtype=np.float32)
        self.onehot = defaultdict(lambda: np.ones(self.ksize, dtype=np.float32) / self.ksize,
                                  {'a': Code_A, 'c': Code_C, 'g': Code_G, 't': Code_U, 'u': Code_U,
                                   'R': zero, '#': zero, 'Y': zero, 'M': zero, 'K': zero, 'S': zero, 'W': zero,
                                   'B': zero, 'V': zero, 'D': zero, 'N': zero})

    def encode(self, seq):
        seq = [self.onehot[s] for s in seq.lower()]
        seq = np.vstack(seq)
        return seq.transpose()