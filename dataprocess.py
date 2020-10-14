import numpy as np
import torch
from collections import defaultdict
from torch_scatter import scatter_add


class Data(object):
    def __init__(self, x, y, adj, train_mask, val_mask, test_mask):
        self.x = x
        self.y = y
        self.num_nodes = x.shape[0]
        self.adj = adj
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.adj = self.adj.to(device)


# sample partial edges
def sample_edge(edge_index, ratio, seed=123):
    # sample from half side of the symmetric adjacency matrix
    half_edge_index = []
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] < edge_index[1, i]:
            half_edge_index.append(edge_index[:, i].view(2,-1))
    half_edge_index = torch.cat(half_edge_index, dim=1)
    np.random.seed(seed)
    num_edge = half_edge_index.shape[1]
    samples = np.random.choice(num_edge, size=int(ratio*num_edge), replace=False)
    sampled_edge_index = half_edge_index[:, samples]
    sampled_edge_index = torch.cat([sampled_edge_index, sampled_edge_index[torch.LongTensor([1,0])]], dim=1)
    return sampled_edge_index


def random_split(data, data_seed, args):
    if data_seed == -1:
        print("fixed split")
        return 0
    if args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        if not args.keep_train_num:
            num_nodes = data.train_mask.shape[0]
            train_num = torch.sum(data.train_mask)
            val_num = torch.sum(data.val_mask)
            test_num = torch.sum(data.test_mask)
            print(train_num, val_num, test_num)
            data.train_mask = torch.zeros(num_nodes).type(torch.uint8)
            data.val_mask = torch.zeros(num_nodes).type(torch.uint8)
            data.test_mask = torch.zeros(num_nodes).type(torch.uint8)
            inds = np.arange(num_nodes)
            np.random.seed(data_seed)
            np.random.shuffle(inds)
            inds = torch.tensor(inds)
            data.train_mask[inds[:train_num]] = 1
            data.val_mask[inds[train_num:train_num + val_num]] = 1
            data.test_mask[inds[train_num + val_num:train_num + val_num + test_num]] = 1
        else:
            index = data.y
            num_nodes = data.y.shape[0]
            val_num = torch.sum(data.val_mask)
            test_num = torch.sum(data.test_mask)
            input = torch.ones(num_nodes)
            out = scatter_add(input, index)
            train_node_per_class = args.train_num
            data.train_mask = torch.zeros(num_nodes).type(torch.uint8)
            data.val_mask = torch.zeros(num_nodes).type(torch.uint8)
            data.test_mask = torch.zeros(num_nodes).type(torch.uint8)
            inds = np.arange(num_nodes)
            np.random.seed(data_seed)
            np.random.shuffle(inds)
            inds = torch.tensor(inds)
            data_y_shuffle = data.y[inds]
            for i in range(out.shape[0]):
                inds_i = inds[data_y_shuffle==i]
                data.train_mask[inds_i[:train_node_per_class]] = 1
            val_test_inds = np.arange(num_nodes)[data.train_mask.numpy() == 0]
            np.random.shuffle(val_test_inds)
            val_test_inds = torch.tensor(val_test_inds)
            data.val_mask[val_test_inds[:val_num]] = 1
            data.test_mask[val_test_inds[val_num:val_num + test_num]] = 1
    elif args.dataset in ["CoraFull", "Computers", "Photo", "CS"]:
        index = data.y
        num_nodes = data.y.shape[0]
        input = torch.ones(num_nodes)
        out = scatter_add(input, index)
        train_node_per_class = args.train_num
        val_node_per_class = 30
        data.train_mask = torch.zeros(num_nodes).type(torch.uint8)
        data.val_mask = torch.zeros(num_nodes).type(torch.uint8)
        data.test_mask = torch.zeros(num_nodes).type(torch.uint8)
        inds = np.arange(num_nodes)
        np.random.seed(data_seed)
        np.random.shuffle(inds)
        inds = torch.tensor(inds)
        data_y_shuffle = data.y[inds]
        for i in range(out.shape[0]):
            if out[i] <= train_node_per_class + val_node_per_class:
                continue
            inds_i = inds[data_y_shuffle==i]
            data.train_mask[inds_i[:train_node_per_class]] = 1
            data.val_mask[inds_i[train_node_per_class:train_node_per_class+val_node_per_class]] = 1
            data.test_mask[inds_i[train_node_per_class+val_node_per_class:]] = 1


def dataloader(data, Adj, batch_size, sparse=False, shuffle=False):
    inds = np.arange(data.num_nodes)
    if shuffle:
        np.random.shuffle(inds)
    inds = torch.tensor(inds)
    batch_data_list = []
    start = 0
    while start < data.num_nodes:
        end = min(start + batch_size, data.num_nodes)
        batch_inds = inds[start:end]
        if not sparse:
            batch_data = Data(x=data.x[batch_inds], y=data.y[batch_inds],
                    adj=Adj[batch_inds][:, batch_inds],
                    train_mask=data.train_mask[batch_inds],
                    val_mask=data.val_mask[batch_inds],
                    test_mask=data.test_mask[batch_inds])
        else:
            if len(batch_inds) == data.num_nodes:
                inds_value = torch.zeros(data.num_nodes).type(torch.LongTensor)
                inds_value[batch_inds] = torch.arange(len(batch_inds))
                indices = inds_value[Adj]
                values = torch.ones(indices.shape[1])
                sample_adj = torch.sparse.FloatTensor(indices, values, [len(batch_inds), len(batch_inds)])
                batch_data = Data(x=data.x[batch_inds], y=data.y[batch_inds],
                        adj=sample_adj.coalesce(),
                        train_mask=data.train_mask[batch_inds],
                        val_mask=data.val_mask[batch_inds],
                        test_mask=data.test_mask[batch_inds])
            else:
                inds_value = (torch.zeros(data.num_nodes) - data.num_nodes).type(torch.LongTensor)
                inds_value[batch_inds] = torch.arange(len(batch_inds))
                Adj_inds = inds_value[Adj]
                indices = Adj_inds[:, torch.sum(Adj_inds, dim=0) >= 0]
                values = torch.ones(indices.shape[1])
                sample_adj = torch.sparse.FloatTensor(indices, values, [len(batch_inds), len(batch_inds)])
                batch_data = Data(x=data.x[batch_inds], y=data.y[batch_inds],
                        adj=sample_adj.coalesce(),
                        train_mask=data.train_mask[batch_inds],
                        val_mask=data.val_mask[batch_inds],
                        test_mask=data.test_mask[batch_inds])

        batch_data_list.append(batch_data)
        start += batch_size
    return batch_data_list
