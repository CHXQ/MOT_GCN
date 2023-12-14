
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from ipdb import set_trace

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()
    def forward(self, features, A ):
        x = torch.bmm(A, features)
        return x 

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(in_dim *2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)
        agg_feats = self.agg(features,A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out  

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(32)
        self.conv1 = GraphConv(32, 64, MeanAggregator)
        self.conv2 = GraphConv(64, 128, MeanAggregator)
        self.conv3 = GraphConv(128, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256,MeanAggregator)
        
        self.conf_mlp = nn.Sequential(
                            nn.Linear(256, 256),
                            nn.ReLU(256),
                            nn.Linear(256, 1))

        self.node_mlp = nn.Sequential(
            nn.Linear(13, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
    
    def forward(self, input, A):
        
        if torch.isnan(input).any():
            set_trace()
        B, N, D = input.shape
        x = input.view(-1, D)
        x = self.node_mlp(x)
        if torch.isnan(x).any():
            set_trace()

        x = x.view(B, N, -1)
        B, N, D = x.shape
        
        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B, N, D)
        if torch.isnan(x).any():
            set_trace()

        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)
        
        conf = self.conf_mlp(x)
        return x, conf