import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def outer_cat(vector):
    #vector = vector.detach().numpy()
    batch = vector.shape[0]
    L = vector.shape[1]
    d = vector.shape[2]
    final = torch.zeros((batch,L,L,d*2))
    #print(final.shape[0])
    for k in range(batch):
        for i in range(L):
            for j in range(L):
                final[k][i][j] = torch.cat((vector[k][i],vector[k][j]), dim=0)
    final = final.to(device)
    return final
