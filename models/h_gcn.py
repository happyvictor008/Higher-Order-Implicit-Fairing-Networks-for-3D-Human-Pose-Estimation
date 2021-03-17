from __future__ import absolute_import

import torch.nn as nn
from models.h_graph_conv import HGraphConv
import torch 













class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = HGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim*4)
        self.relu = nn.ReLU()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim*4, output_dim, p_dropout)

    def forward(self, x):
        residual = x 
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class HGCN(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4,nodes_group=None, p_dropout=None):
        super(HGCN, self).__init__()

        _gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout)]  #First hgcn
        _gconv_layers = []
        hid_dim = hid_dim * 4

        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim//4, hid_dim//4, p_dropout=p_dropout))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = HGraphConv(hid_dim, coords_dim[1], adj)
    def forward(self, x): 
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        
        return out



