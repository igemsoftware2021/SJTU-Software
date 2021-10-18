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
    dd = "./dd "+filename
    a = os.popen(dd).read()
    score = float(a)
    return score

if __name__ == "__main__":
    seqlist = []
    seqlist.append("ATGGCTTTAA")
    seqlist.append("TTAAAGCCAT")
    seqlist.append("TTAAGCCA")
    score = spurious_hybrid(3,seqlist)
    print(score)