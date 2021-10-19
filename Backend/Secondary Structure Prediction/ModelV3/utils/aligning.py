import numpy as np

def bp_align(bp, size, max_len):

    blank = '#'
    align_bp = [0 for x in range(size)]

    for i in range(size):
        align_bp[i] = bp[i] + blank * (max_len - len(bp[i]))

    return align_bp


def target_align(target, size, max_len):

    label_blank = '#'
    align_target = [0 for x in range(size)]

    for i in range(size):
        align_target[i] = target[i] + label_blank * (max_len - len(target[i]))

    return align_target

def encodecp_align(wv_bp, size, dim, max_len):

    zero = np.zeros(dim, dtype=np.float32)

    for i in range(size):
        length = len(wv_bp[i])
        wv_bp[i][length:max_len] = zero

    return wv_bp

def de_align(align_seq, align_target, size, length):

    seq = [size]
    target = [size]

    # for i in range(size):
    #     if len(seq[i]) > max_len:
    #         max_len = len(seq[i])

    for i in range(size):
            seq[i] = align_seq[i][:length]
            target[i] = align_target[i][:length]

    return seq, target

