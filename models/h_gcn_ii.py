from __future__ import absolute_import

import torch.nn as nn
from models.h_graph_conv_ii import HGraphConvII
import torch 













class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None,l=None):
        super(_GraphConv, self).__init__()

        self.gconv = HGraphConvII(input_dim, output_dim, adj,l)
        self.bn = nn.BatchNorm1d(output_dim*4)
        self.relu = nn.ReLU()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, x0):

        x = self.gconv(x, x0).transpose(1, 2)##################
        x = self.bn(x).transpose(1, 2)
        
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x



class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout,l):
        super(_ResGraphConv, self).__init__()




        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout,l+1)
        self.gconv2 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+2)
        self.gconv3 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+3)
        self.gconv4 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+4)
        self.gconv5 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+5)
        self.gconv6 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+6)
        self.gconv7 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+7)
        self.gconv8 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+8)
        self.gconv9 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+9)
        self.gconv10 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+10)
        self.gconv11 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+11)
        self.gconv12 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+12)
        self.gconv13 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+13)
        self.gconv14 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+14)
        self.gconv15 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout,l+15)
        self.gconv16 = _GraphConv(adj, hid_dim*4, output_dim, p_dropout,l+8)#13

    def forward(self, x,x_0):
        initial = x_0
        #print('initial========',initial.size())
        #print('x========',x.size())
        out = self.gconv1(x,initial)
        out = self.gconv2(out,initial)
        out = self.gconv3(out,initial)
        out = self.gconv4(out,initial)



        out = self.gconv5(out,initial)
        out = self.gconv6(out,initial)
        out = self.gconv7(out,initial)
        #out = self.gconv8(out,initial)
        #out = self.gconv9(out,initial)
        #out = self.gconv10(out,initial)
        #out = self.gconv11(out,initial)
        #out = self.gconv12(out,initial)
        #out = self.gconv13(out,initial)
        #out = self.gconv14(out,initial)
        #out = self.gconv15(out,initial)
        out = self.gconv16(out,initial)

        return  out


class HGCNII(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4,nodes_group=None, p_dropout=None):
        super(HGCNII, self).__init__()

        #_gconv_input = [_GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout,l=1)]  #First hgcn
        #_gconv_layers = []
        #hid_dim = hid_dim * 4

        #for i in range(num_layers):
            #layer_index=2*i+2
            #_gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim//4, hid_dim//4, p_dropout=p_dropout,l=layer_index))

        #self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_input = _GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout,l=1)
        self.gconv_layers = _ResGraphConv(adj, hid_dim*4, hid_dim, hid_dim, p_dropout=p_dropout,l=1)
        #self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = HGraphConvII(hid_dim*4, coords_dim[1], adj,l=1)#layer_index+2
    def forward(self, x): 

        out = self.gconv_input(x,x)
        x_0=out
        #print("xxxxxxxxxxxxxxxxx0000000000000000000",x_0.size())
        out = self.gconv_layers(out,x_0)
        out = self.gconv_output(out,x_0)
        
        return out



