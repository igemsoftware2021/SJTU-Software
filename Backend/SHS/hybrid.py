import os

def spurious_hybrid(num,seqlist):
    filename = "cachefile"
    score = 0
    with open(filename,'w') as fp:
        fp.write(str(num)+'\n')
        fp.write(">targetseq"+'\n')
        fp.write(seqlist[0]+'\n')
        for i in range(num-1):
            fp.write(">otherseq"+'\n')
            fp.write(seqlist[i+1]+'\n')
    dd = "./dd_new "+filename
    a = os.popen(dd).read()
    score = float(a)
    return score

if __name__ == "__main__":
    seqlist = []
    seqlist.append("ATGGCTTTAA")
    seqlist.append("TTAAAGCCAT")
    seqlist.append("TTAAGCCA")
    score1 = spurious_hybrid(3,seqlist)

    for i in range(len(seqlist)-1):
        seqlist[i+1] = seqlist[i+1][::-1]
    score2 = spurious_hybrid(3,seqlist)

    print(score1 if score1>score2 else score2)