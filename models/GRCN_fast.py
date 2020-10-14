import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, knn_graph
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from .model_utils import GCNConv_diag, GCNConv_dense, EOS
import torch_sparse as ts


class Model(torch.nn.Module):

    def __init__(self, num_nodes, num_features, num_classes, device, args):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.graph_nhid = int(args.hid_graph.split(":")[0])
        self.graph_nhid2 = int(args.hid_graph.split(":")[1])
        self.nhid = args.nhid
        self.conv1 = GCNConv_dense(num_features, self.nhid)
        self.conv2 = GCNConv_dense(self.nhid, num_classes)
        if args.layertype == "dense":
            self.conv_graph = GCNConv_dense(num_features, self.graph_nhid)
            self.conv_graph2 = GCNConv_dense(self.graph_nhid, self.graph_nhid2)
        elif args.layertype == "diag":
            self.conv_graph = GCNConv_diag(num_features, device)
            self.conv_graph2 = GCNConv_diag(num_features, device)
        else:
            exit("wrong layer type")
        self.F = args.F
        self.F_graph = args.F_graph
        self.dropout = args.dropout
        self.K = args.compl_param.split(":")[0]
        self.mask = None
        self.Adj_new = None
        self._normalize = args.normalize
        self.device = device
        self.reduce = args.reduce
        self.sparse = args.sparse
        self.norm_mode = "sym"
        self.topindex = None

    def init_para(self):
        self.conv1.init_para()
        self.conv2.init_para()
        self.conv_graph.init_para()
        self.conv_graph2.init_para()

    def graph_parameters(self):
        return list(self.conv_graph.parameters()) + list(self.conv_graph2.parameters())

    def base_parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    def cal_similarity_graph(self, node_embeddings):
        # similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
        similarity_graph = torch.mm(node_embeddings[:, :int(self.num_features/2)], node_embeddings[:, :int(self.num_features/2)].t())
        similarity_graph += torch.mm(node_embeddings[:, int(self.num_features/2):], node_embeddings[:, int(self.num_features/2):].t())
        # node_embeddings_top = node_embeddings[self.topindex]
        # similarity_graph = node_embeddings.unsqueeze(1) * node_embeddings_top
        # similarity_graph = torch.sum(similarity_graph, dim=-1)
        return similarity_graph

    def normalize(self, adj, mode="sym" ,sparse=False):
        if not sparse:
            if mode == "sym":
                inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
                return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
            elif mode == "row":
                inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
                return inv_degree[:, None] * adj
            else:
                exit("wrong norm mode")
        else:
            adj = adj.coalesce()
            if mode == "sym":
                inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + EOS)
                D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
            elif mode == "row":
                inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
                D_value = inv_degree[adj.indices()[0]]
            else:
                exit("wrong norm mode")
            new_values = adj.values() * D_value
            return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).to(self.device)

    def _sparse_graph(self, raw_graph, K, sparse):
        if self.reduce == "knn":
            if self.topindex is None:
                values, indices = raw_graph.topk(k=int(K), dim=-1)
                self.topindex = indices
                assert torch.max(indices) < raw_graph.shape[1]
            else:
                indices = self.topindex
                values = raw_graph[torch.arange(raw_graph.shape[0]).view(-1,1), indices]
            assert torch.sum(torch.isnan(values)) == 0
            if not sparse:
                self.mask = torch.zeros(raw_graph.shape).to(self.device)
                self.mask[torch.arange(raw_graph.shape[0]).view(-1,1), indices] = 1.
                self.mask[indices, torch.arange(raw_graph.shape[1]).view(-1,1)] = 1.
            else:
                inds = torch.stack([torch.arange(raw_graph.shape[0]).view(-1,1).expand(-1,int(K)).contiguous().view(1,-1)[0].to(self.device),
                                     indices.view(1,-1)[0]])
                inds = torch.cat([inds, torch.stack([inds[1], inds[0]])], dim=1)
                values = torch.cat([values.view(1,-1)[0], values.view(1,-1)[0]])
                return inds, values
        else:
            exit("wrong sparsification method")
        self.mask.requires_grad = False
        sparse_graph = raw_graph * self.mask
        return sparse_graph

    def _node_embeddings(self, input, Adj, sparse=False):
        norm_Adj = self.normalize(Adj, self.norm_mode, sparse)
        if self.F_graph != "identity":
            node_embeddings = self.F_graph(self.conv_graph(input, norm_Adj, sparse))
            node_embeddings = self.conv_graph2(node_embeddings, norm_Adj, sparse)
        else:
            node_embeddings = self.conv_graph(input, norm_Adj, sparse)
            node_embeddings = self.conv_graph2(node_embeddings, norm_Adj, sparse)
        if self._normalize:
            node_embeddings = F.normalize(node_embeddings, dim=1, p=2)
        return node_embeddings

    def forward(self, input, Adj):
        Adj.requires_grad = False
        node_embeddings = self._node_embeddings(input, Adj, self.sparse)
        Adj_new = self.cal_similarity_graph(node_embeddings)

        if not self.sparse:
            Adj_new = self._sparse_graph(Adj_new, self.K, self.sparse)
            Adj_new = self.normalize(Adj + Adj_new, self.norm_mode)
        else:
            Adj_new_indices, Adj_new_values = self._sparse_graph(Adj_new, self.K, self.sparse)
            new_inds = torch.cat([Adj.indices(), Adj_new_indices], dim=1)
            new_values = torch.cat([Adj.values(), Adj_new_values])
            Adj_new = torch.sparse.FloatTensor(new_inds, new_values, Adj.size()).to(self.device)
            Adj_new = self.normalize(Adj_new, self.norm_mode, self.sparse)

        x = self.conv1(input, Adj_new, self.sparse)
        x = F.dropout(self.F(x), training=self.training, p=self.dropout)
        x = self.conv2(x, Adj_new, self.sparse)

        return F.log_softmax(x, dim=1)
