import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from utils.GraphTools import *

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, batch_size, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(batch_size, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(batch_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.bmm(input, self.weight)
        # s = torch.isnan(input)
        # for i in range(1208):
        #    for j in range(64):
        #         if s[i][j] == True:
        #             print(1)
        # print('ok')
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias.view(self.bias.shape[0], 1 , self.bias.shape[1])
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)


class GNN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid_decode1, dropout, batch_size):
        super(GNN, self).__init__()

        # original graph
        self.gc1 = GraphConvolution(nfeat, nhid1, batch_size)
        self.gc2 = GraphConvolution(nhid1, nhid2, batch_size)

        self.dropout = dropout

        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        # feat_p1 = x[idx[0]]  # the first biomedical entity embedding retrieved
        # feat_p2 = x[idx[1]]  # the second biomedical entity embedding retrieved
        # feat = torch.cat((feat_p1, feat_p2), dim=1)
        # o = self.decoder1(feat)
        # o = self.decoder2(o)
        return x

class GNNC(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nhid4, dropout, batch_size):
        super(GNNC, self).__init__()

        # original graph
        self.gc1 = GraphConvolution(nfeat, nhid1, batch_size)
        self.gc2 = GraphConvolution(nhid1, nhid2, batch_size)
        self.gc3 = GraphConvolution(nhid2, nhid3, batch_size)
        self.gc4 = GraphConvolution(nhid3, nhid4, batch_size)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)

        return x