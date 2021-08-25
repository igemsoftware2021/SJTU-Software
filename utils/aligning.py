def bp_align(seq, target, size, max_len=150):

    blank = '0'

    # for i in range(size):
    #     if len(seq[i]) > max_len:
    #         max_len = len(seq[i])

    align_seq = [0 for x in range(size)]
    align_target = [0 for x in range(size)]

    for i in range(size):
        align_seq[i] = seq[i] + blank * (max_len - len(seq[i]))
        align_target[i] = target[i] + blank * (max_len - len(target[i]))

    return align_seq, align_target

