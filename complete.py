'''
Complete the graph structrue
Be careful that the completed graph should be symmetric
Be careful that you need to renormalize the completed adjacency matrix
'''
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import *
from collections import Counter

import numpy as np
import random

EOS = 1e-10

# construct adj matrix from edge_index
def convert_edge2adj(edge_index, num_nodes):
    # float type
    mat = torch.zeros((num_nodes, num_nodes))
    for i in range(edge_index.shape[1]):
        x, y = edge_index[:, i]
        mat[x, y] = mat[y, x] = 1
    return mat

# generate normalized random walk adjacency matrix
def normalize(adj):
    inv_sqrt_degree = 1. / torch.sqrt(adj.sum(dim=1, keepdim=False))
    inv_sqrt_degree[inv_sqrt_degree == float("Inf")] = 0
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]

# transform flat index to adjacency edge index for input of GraphConv
def flat_index2adj_index(indices, num_nodes):
    size = len(indices)
    ret_list = []
    for num, index in enumerate(indices):
        # TODO: check whether the transformation is correct or not
        ret_list.append([index / num_nodes, index % num_nodes])
    ret_indices = torch.tensor(ret_list)
    return ret_indices.t().contiguous()

def sparse_complete(true_adj, pred_adj, k):
    flat_true_adj = true_adj.view(-1)
    # remove diagonal entries
    flat_pred_adj = (pred_adj).view(-1)
    sorted, indices = torch.topk(flat_pred_adj, k)
    two_d_indices = flat_index2adj_index(indices, pred_adj.shape[0])
    return 1.0 * flat_true_adj[indices].sum() / k, two_d_indices

def _complete_acc(true_adj, pred_adj):
    flat_true_adj = true_adj.view(-1)
    pos_index = (flat_true_adj == 1).nonzero().view(-1)
    neg_index = (flat_true_adj == 0).nonzero().view(-1)
    pos_num = pos_index.shape[0]
    neg_num = neg_index.shape[0]
    sample_neg_num = pos_num * 5
    neg_sample = neg_index[torch.randperm(neg_num)[:sample_neg_num]]
    flat_pred_adj = (pred_adj).view(-1)
    total_index = torch.cat([pos_index, neg_sample])
    sorted, indices = torch.topk(flat_pred_adj[total_index], pos_num)
    return 1.0 * flat_true_adj[total_index[indices]].sum() / pos_num

def cal_similarity_graph(node_features):
    similarity_graph = torch.mm(node_features, node_features.t())
    return similarity_graph


class Complete(object):

    def __init__(self, sampled_edge_index, data, model, device, args):
        super(Complete, self).__init__()
        self.sampled_edge_index = sampled_edge_index.to(device)
        self.data = data.to(device)
        self.device = device
        self.model = model
        self.dense = args.dense
        self.reduce = args.reduce

        # pre-preprocessing of the edge set as normalized adjacency graphs, for the ease of edge completion
        if self.dense:
            self.adj_full = convert_edge2adj(data.edge_index, data.num_nodes).to(device)
            self.adj_part = convert_edge2adj(sampled_edge_index, data.num_nodes).to(device)
            self.loop_adj_part = torch.eye(self.adj_part.shape[0]).to(device) + self.adj_part
            self.norm_adj_part = normalize(self.loop_adj_part)
        else:
            loop_edge_index = torch.stack([torch.arange(data.num_nodes), torch.arange(data.num_nodes)])
            self.loop_adj_part = torch.cat([sampled_edge_index, loop_edge_index], dim=1)

    def complete_graph(self, method, compl_param):
        self.model.eval()
        if method in ["GCN", "SGC", "GAT"]:
            if self.dense:
                return 0., self.norm_adj_part
            else:
                return 0., self.sampled_edge_index
        elif method in ["GAT_official", "GAT_dense"]:
            if self.dense:
                return 0., self.norm_adj_part
            else:
                return 0., self.loop_adj_part
        else:
            exit("wrong model")
