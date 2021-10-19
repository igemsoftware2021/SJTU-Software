import torch
# 接受一个str, 返回一个tensor类型的上三角邻接矩阵
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def dash2matrix(str,length):
    # seq_len = len(str)
    lst = []
    labels = []
    mat = torch.zeros((length,length))
    for i in range(len(str)):
        #print(lst)
        if(str[i]=='('):
            lst.append(i)
        elif(str[i] == '.'):
            pass
        elif(str[i] == ')'):
            idx = lst.pop()
            mat[idx,i] = 1
            mat[i,idx] = 1

    return mat

# 创建邻接矩阵和idx_map
def buildAdjandIdxmap(adj,length):
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # idx = range(length)
    # idx_map = {j: i for i, j in enumerate(idx)}

    return adj

def build_idx(length):
    idx1 = []
    idx2 = []
    for i in range(length):
        for j in range(i, length):
            idx1.append(i)
            idx2.append(j)
    idx1 = torch.tensor(idx1)
    idx2 = torch.tensor(idx2)
    inp = (idx1, idx2)
    inp = list(inp)

    return inp