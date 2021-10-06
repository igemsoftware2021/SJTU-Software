import torch
import os

def file2matrix(filename,length):
    mat = torch.zeros((length,length))
    with open(filename,'r') as fp:
        a = fp.readlines()
        for item in a:
            b = item.split()
            if(len(b)!=0):
                print(b)
                i = int(b[0])-1
                j = int(b[1])-1
                num = float(b[2])
                mat[i][j] = num
    return mat

def getadj(seq,length):
    filename = 'cache.txt'
    LP = './LinearPartition/linearpartition -V -o '
    LP_get = 'echo ' + seq + ' | ' + LP + filename
    if(os.path.exists(filename)):
        os.remove(filename)
    os.system(LP_get)
    mat = file2matrix(filename,length)
    return mat