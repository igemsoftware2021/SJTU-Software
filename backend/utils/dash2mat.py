import torch

# 接受一个str, 返回一个tensor类型的上三角邻接矩阵
def dash2matrix(str,length):
    # seq_len = len(str)
    lst = []
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

# str1 = "..((..))(.)"
# mat1 = dash2matrix(str1)
# print(mat1.shape,mat1)


