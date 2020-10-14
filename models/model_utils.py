import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, knn_graph
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


EOS = 1e-10

class GCNConv_dense(torch.nn.Module):
    '''
    A GCN convolution layer of dense matrix multiplication
    '''
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output


class Linear(torch.nn.Module):
    '''
    A GCN convolution layer of dense matrix multiplication
    '''
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        return hidden


class Diag(torch.nn.Module):
    '''
    A GCN convolution layer of diagonal matrix multiplication
    '''
    def __init__(self, input_size, device):
        super(Diag, self).__init__()
        self.W = torch.nn.Parameter(torch.ones(input_size).to(device))
        self.input_size = input_size

    def init_para(self):
        self.W = torch.nn.Parameter(torch.ones(self.input_size).to(device))

    def forward(self, input, A, sparse=False):
        hidden = input @ torch.diag(self.W)
        return hidden


class GCNConv_diag(torch.nn.Module):
    '''
    A GCN convolution layer of diagonal matrix multiplication
    '''
    def __init__(self, input_size, device):
        super(GCNConv_diag, self).__init__()
        self.W = torch.nn.Parameter(torch.ones(input_size).to(device))
        # inds = torch.stack([torch.arange(input_size), torch.arange(input_size)]).to(device)
        # self.mW = torch.sparse.FloatTensor(inds, self.W, torch.Size([input_size,input_size]))
        self.input_size = input_size

    def init_para(self):
        self.W = torch.nn.Parameter(torch.ones(self.input_size).to(device))

    def forward(self, input, A, sparse=False):
        hidden = input @ torch.diag(self.W)
        # hidden = torch.sparse.mm(self.mW, input.t()).t()
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output
