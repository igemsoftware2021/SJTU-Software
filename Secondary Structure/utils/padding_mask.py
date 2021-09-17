import torch

def padding_mask(seq, max_len):
    # seq的形状都是[B,L]
    # `PAD` is #
    pad_mask = seq.eq(4)
    pad_mask = pad_mask.view(-1, max_len)  # shape [B, L_q]

    return pad_mask