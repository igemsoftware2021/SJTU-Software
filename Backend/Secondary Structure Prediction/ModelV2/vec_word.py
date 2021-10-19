from gensim.models import word2vec
import logging
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from Getdataset import *

batch_size = 256
global_length = 500

trainSet = IntelDNADataset('./full_data/bpRNA_dataset', './full_data/adj_dataset_full',
                           './full_data/node2vec_dataset_full_16', global_length, 'train')
testSet = IntelDNADataset('./full_data/bpRNA_dataset', './full_data/adj_dataset_full',
                          './full_data/node2vec_dataset_full_16', global_length, 'test')
train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=False, drop_last=True)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
seqs = ()
for data in tqdm(train_dataloader):
    seq, _, _, _, _ = data
    seqs += seq

seqs = list(seqs)

sentences = [list(s) for s in seqs]

model = word2vec.Word2Vec(seqs, size=16, window=12, )

vocabs = model.wv.__getitem__('A')

model.similarity('A', 'U')

model.save('Word.ckpt')

print()