import torch
from torch.utils.data import Dataset
import os
import numpy as np


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

        return seq, result, file2matrix(self.adj_name, self.length), file2vec(self.vec_name), self.bp_name

    def __len__(self):
        return len(self.data_list)
