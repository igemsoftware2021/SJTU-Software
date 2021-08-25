import torch
from torch.utils.data import Dataset
import os


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


class IntelDNADataset(Dataset):
    def __init__(self, data_dir, data_type='train'):
        super(IntelDNADataset, self).__init__()
        self.data_dir = data_dir
        self.sub_dir = 'TR0'
        # Obtain the data type
        if data_type == 'train':
            self.sub_dir = 'TR0'
        if data_type == 'test':
            self.sub_dir = 'TS0'
        if data_type == 'dev':
            self.sub_dir = 'VL0'

        self.path = os.path.join(self.data_dir, self.sub_dir)
        self.data_list = os.listdir(self.path)

    def __getitem__(self, idx):
        seq = []
        result = []
        self.bp_dir = os.path.join(self.path, self.data_list[idx])

        with open(self.bp_dir) as f:
            lines = f.readlines()
            seq = lines[3]
            seq = seq.rstrip('\n')
            result = lines[4]
            result = result.rstrip('\n')

        return seq, result

    def __len__(self):
        return len(self.data_list)
