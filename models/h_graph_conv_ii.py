from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HGraphConvII(nn.Module):
    """
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj,l, bias=True,):
        super(HGraphConvII, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.l = l
        self.alpha_l=0.5#0.5#0.2
        self.lamda=0.5#1.0#0.5
        if self.l==None:
            self.l=1.0
        #l=4#####
        self.beta = math.log(self.lamda/self.l+1.0)
        
        

        

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
        
        #print('beta===========================',self.beta)
        #print('lamda===========================',self.lamda)
        #print('l===========================',self.l)
        else:
            self.register_parameter('bias', None)

    def forward(self, input1, input2):
        #print('Input===========================',input.size())

        
        #input=features

        #h0 = torch.matmul(input, self.W[0])
        #h1 = torch.matmul(input, self.W[1])
        #h2 = torch.matmul(input, self.W[2])
        #h3 = torch.matmul(input, self.W[3])

    
        A_0 = -9e15 * torch.ones_like(self.adj_0).to(input1.device)
        A_1 = -9e15 * torch.ones_like(self.adj_1).to(input1.device) # without self-connection 
        A_2 = -9e15 * torch.ones_like(self.adj_2).to(input1.device)
        A_3 = -9e15 * torch.ones_like(self.adj_3).to(input1.device)
         
        A_0[self.m_0] = self.e_0
        A_1[self.m_1] = self.e_1
        A_2[self.m_2] = self.e_2
        A_3[self.m_3] = self.e_3

        A_0 = F.softmax(A_0, dim=1)
        A_1 = F.softmax(A_1, dim=1)
        A_2 = F.softmax(A_2, dim=1)
        A_3 = F.softmax(A_3, dim=1)


        #print('A_0===========================',A_0.size())



        h0 = torch.matmul((1-self.alpha_l)*A_0,input1)+self.alpha_l*input2
        #print('h_0===========================',h0.size())
        #print('w_0===========================',self.W[0].size())

        #output_0 = torch.matmul(h0,self.beta*self.W[0])
        output_0 = torch.matmul(h0,self.beta*self.W[0])
        #print('+++++++++',h0.size())
        #print('+++++++++++++++',self.W[0].size())
        #print("output0----------",output_0.size())
        #output_0 = h0*(1-beta)+h0_2

        h1 = torch.matmul((1-self.alpha_l)*A_1,input1)+self.alpha_l*input2
        output_1 = torch.matmul(h1,self.beta*self.W[1])
        #h1= torch.matmul(h1,(1-beta)*)
        #h1_2 = torch.matmul(h1,beta*self.W[1])
        #output_1 = h1+h1_2

        h2 = torch.matmul((1-self.alpha_l)*A_2,input1)+self.alpha_l*input2
        output_2 = torch.matmul(h2,self.beta*self.W[2])
        #h2_2 = torch.matmul(h2,beta*self.W[2])
        #output_2 = h2*(1-beta)+h2_2

        #input
        h3 = torch.matmul((1-self.alpha_l)*A_3,input1)+self.alpha_l*input2
        output_3 = torch.matmul(h3,self.beta*self.W[3])
        #h3_2 = torch.matmul(h3,beta*self.W[3])
        #output_3 = h1*(1-beta)+h3_2







        #output_0 = torch.matmul(A_0, h0) 
        #output_1 = torch.matmul(A_1, h1) 
        #output_2 = torch.matmul(A_2, h2) 
        #output_3 = torch.matmul(A_3, h3)

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
