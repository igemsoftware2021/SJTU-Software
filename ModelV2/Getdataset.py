import re
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import os
import numpy as np
from gensim.models import word2vec

# 接受一个str, 返回一个tensor类型的上三角邻接矩阵
def dash2matrix(str, length):
    # seq_len = len(str)
    lst = []
    mat = torch.zeros((length, length))
    for i in range(len(str)):
        # print(lst)
        if (str[i] == '('):
            lst.append(i)
        elif (str[i] == '.'):
            pass
        elif (str[i] == ')'):
            idx = lst.pop()
            mat[idx, i] = 1
            mat[i, idx] = 1
    return mat


# Load node2vec from filename
def file2vec(filename):
    vec = np.load(filename)
    return vec


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


# Load adj_matrix from filename
def file2matrix(filename, length):
    mat = torch.zeros((length, length))
    with open(filename, 'r') as fp:
        a = fp.readlines()
        for item in a:
            b = item.split()
            if (len(b) != 0):
                i = int(b[0]) - 1
                j = int(b[1]) - 1
                num = float(b[2])
                mat[i][j] = num
    return mat


class IntelDNADataset(Dataset):
    def __init__(self, data_dir, adj_dir, vec_dir, length, data_type='train'):
        super(IntelDNADataset, self).__init__()

        # base pair data
        self.data_dir = data_dir
        # adj data
        self.adj_dir = adj_dir
        # vec data
        self.vec_dir = vec_dir

        self.length = length
        self.sub_dir = data_type
        self.encode_size = 12

        # Obtain the data type
        if data_type == 'train':
            self.sub_dir = 'TR0'
        if data_type == 'test':
            self.sub_dir = 'TS0'
        if data_type == 'dev':
            self.sub_dir = 'VL0'

        # Create base pair file path
        self.path = os.path.join(self.data_dir, self.sub_dir)
        # Create adj_matrix file path
        self.adj_path = os.path.join(self.adj_dir, self.sub_dir)
        # Create node2vec file path
        self.vec_path = os.path.join(self.vec_dir, self.sub_dir)

        # Create filename list
        self.data_list = os.listdir(self.path)
        self.adj_list = os.listdir(self.adj_path)
        self.vec_list = os.listdir(self.vec_path)

    def __getitem__(self, idx):

        # Obtain filename via idx
        self.bp_name = os.path.join(self.path, self.data_list[idx])
        self.adj_name = os.path.join(self.adj_path, self.adj_list[idx])
        self.vec_name = os.path.join(self.vec_path, self.vec_list[idx])

        with open(self.bp_name) as f:
            lines = f.readlines()
            start_line = 0

            length = re.findall("\d+",lines[1])

            lengths = int(length[0])

            for i in range(len(lines)):
                if lines[i].find('#') != -1:
                    continue
                else:
                    start_line = i
                    break

            seq = lines[start_line]
            seq = seq.rstrip('\n')
            result = lines[start_line + 1]
            result = result.rstrip('\n')

            model = word2vec.Word2Vec(seq, window=16, size=self.encode_size)

            Code_A = model.__getitem__("A")
            Code_U = model.__getitem__("U")
            Code_C = model.__getitem__("C")
            Code_G = model.__getitem__("G")

            zero = np.zeros(self.encode_size, dtype=np.float32)
            self.onehot = defaultdict(lambda: np.ones(self.ksize, dtype=np.float32) / self.ksize,
                                      {'A': Code_A, 'C': Code_C, 'G': Code_G, 'T': Code_U, 'U': Code_U,
                                       'R': zero, '#': zero, 'Y': zero, 'M': zero, 'K': zero, 'S': zero, 'W': zero,
                                       'B': zero, 'V': zero, 'D': zero, 'N': zero})

            encode_seq = [self.onehot[s] for s in seq.upper()]
            encode_seq = np.vstack(encode_seq)
            encode_seq.transpose(encode_seq)


        return seq, encode_seq, result, file2matrix(self.adj_name, self.length), lengths

    def __len__(self):
        return len(self.data_list)
