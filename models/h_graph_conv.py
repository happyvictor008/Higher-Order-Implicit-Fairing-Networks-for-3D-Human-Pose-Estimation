from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HGraphConv(nn.Module):
    """
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(HGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(4, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
  
        self.adj_0 = torch.eye(adj.size(0), dtype=torch.float) # declare self-connections
        self.m_0 = (self.adj_0 > 0)
        
        self.e_0 =  nn.Parameter(torch.zeros(1, len(self.m_0.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_0.data, 1)

        self.adj_1 = adj # one_hop neighbors
        self.m_1 = (self.adj_1 > 0)
        self.e_1 = nn.Parameter(torch.zeros(1, len(self.m_1.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_1.data, 1)
        
        self.adj_2 = torch.matmul(self.adj_1, adj) # two_hop neighbors
        self.m_2 = (self.adj_2 > 0)
        self.e_2 = nn.Parameter(torch.zeros(1, len(self.m_2.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_2.data, 1)

        self.adj_3 = torch.matmul(self.adj_2, adj) # three_hop neighbors
        self.m_3 = (self.adj_3 > 0)
        self.e_3 = nn.Parameter(torch.zeros(1, len(self.m_3.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e_3.data,1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features*4, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
            
            self.bias_2 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias_2.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        #print(input.size())
        h0 = torch.matmul(input, self.W[0])#h*w
        h1 = torch.matmul(input, self.W[1])
        h2 = torch.matmul(input, self.W[2])
        h3 = torch.matmul(input, self.W[3])
        #print(input.size())
        #print(self.W[0].size())
        
        

    
        A_0 = -9e15 * torch.ones_like(self.adj_0).to(input.device)
        A_1 = -9e15 * torch.ones_like(self.adj_1).to(input.device) # without self-connection 
        A_2 = -9e15 * torch.ones_like(self.adj_2).to(input.device)
        A_3 = -9e15 * torch.ones_like(self.adj_3).to(input.device)

         
        A_0[self.m_0] = self.e_0
        A_1[self.m_1] = self.e_1
        A_2[self.m_2] = self.e_2
        A_3[self.m_3] = self.e_3

        
        #print('m_0',self.m_0)m_0 k-hop 邻接矩阵位置 

        A_0 = F.softmax(A_0, dim=1)
        A_1 = F.softmax(A_1, dim=1)
        A_2 = F.softmax(A_2, dim=1)
        A_3 = F.softmax(A_3, dim=1)

        #print('A1',A_1)k-hop 邻接矩阵
        #print(A_0.size())
        #print('++++++++++++')
        #print(h0.size())
       #print(A_0.size())
        #print('----------------------')
        



        output_0 = torch.matmul(A_0, h0) 
        output_1 = torch.matmul(A_1, h1) 
        output_2 = torch.matmul(A_2, h2) 
        output_3 = torch.matmul(A_3, h3)
        #print(output_0.size())

        if self.out_features is not 3:  
            output = torch.cat([output_0, output_1, output_2, output_3], dim = 2)####################################################cat
        else: 
            output = output_0 + output_1 + output_2 + output_3
            return output + self.bias_2.view(1,1,-1)
        if self.bias is not None:
            return output + self.bias.view(1,1,-1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
