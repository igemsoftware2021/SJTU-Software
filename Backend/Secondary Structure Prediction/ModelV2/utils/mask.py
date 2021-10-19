import torch
import numpy as np

def padding_mask(seq, max_len):
    # seq的形状都是[B,L]
    # `PAD` is #
    pad_mask = seq.eq(4)
    pad_mask = pad_mask.view(-1, max_len)  # shape [B, L_q]

    return pad_mask

def contact_map_masks(seq_lens, max_len):
    n_seq = len(seq_lens)
    masks = np.zeros([n_seq, max_len, max_len])
    for i in range(n_seq):
        l = int(seq_lens[i].cpu().numpy())
        masks[i, :l, :l]=1
    return masks