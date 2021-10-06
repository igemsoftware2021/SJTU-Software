import torch

# 接受一个str, 返回一个L*3的标签
def dash2label(str,length):
    mat = torch.zeros((length,3))
    for i in range(len(str)):
        if(str[i]=='('):
            mat[i] = torch.tensor((1,0,0))
        elif(str[i] == '.'):
            mat[i] = torch.tensor((0,1,0))
        elif(str[i] == ')'):
            mat[i] = torch.tensor((0,0,1))
    return mat

def label2dash(out):
    dash = []
    for i in range(len(out)):
        if(out[i]==0):
            dash.append('(')
        elif(out[i]==1):
            dash.append('.')
        elif(out[i]==2):
            dash.append(')')
    return dash

# 接受一个str, 返回一个L*2的标签
def dash2label_binary(str,length):
    mat = torch.zeros((length,2))
    for i in range(len(str)):
        if(str[i]=='(' or str[i] == ')'):
            mat[i] = torch.tensor((1,0,))
        elif(str[i] == '.'):
            mat[i] = torch.tensor((0,1))
    return mat

# 接受一个str, 返回一个L*1的标签
def dash2label_1d(str,length):
    mat = torch.zeros(length)
    for i in range(len(str)):
        if(str[i]=='(' or str[i] == ')'):
            mat[i] = 1
        elif(str[i] == '.'):
            mat[i] = 0
    return mat

def out2label(tensor,length):
    mat = torch.zeros((tensor.shape[0],length))
    for k in range(tensor.shape[0]):
        for i in range(tensor.shape[1]):
            if (tensor[k][i] >= 0.5):
                mat[k][i] = 1
            elif (tensor[k][i] < 0.5):
                mat[k][i] = 0
    return mat


if __name__ == "__main__":
    out = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
        1, 1, 1, 1]
    pdash = label2dash(out)
    print("".join(pdash))

    for i in range(64):
        print(i)
